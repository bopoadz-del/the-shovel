"""Oracle Aconex REST adapter.

Endpoints are taken from Oracle Aconex public API docs. This client does not
invent paths. It fails closed when ACONEX_ENABLED is off or credentials are
missing — it never fabricates a local RFI/submittal register.

Documented sources (Oracle Help / API News):
  Resource server: https://api.aconex.com
  Lobby token:     POST {LOBBY}/auth/token  (client_credentials or bearer token)
  List Mail:       GET  /api/projects/{projectid}/mail?mail_box=inbox
  View Mail:       GET  /api/projects/{projectid}/mail/{mailid}
                   (same /mail/{mailId} path family as View Mail Reply Schema)
  Create Mail:     POST /api/projects/{projectid}/mail/
                   Content-Type multipart/mixed; XML part
                   application/vnd.aconex.mail.v2+xml
  List Documents:  GET  /api/projects/{projectid}/register
  Document meta:   GET  /api/projects/{projectid}/register/{documentid}/metadata
  Download file:   GET  /api/projects/{projectid}/register/{documentid}
  Mail search:     POST /api/projects/{projectid}/mail/search  (API News May 2026)
"""

from __future__ import annotations

import base64
import logging
import xml.etree.ElementTree as ET
from typing import Any, Optional
from xml.sax.saxutils import escape as xml_escape

import httpx

from app.core.cde import config
from app.core.cde.errors import CdeError, CdeNotConfiguredError
from app.core.cde.protocol import CdeClient
from app.core.cde.types import CdeDocument, CdeMail, CdeMailDraft, CdePostResult

logger = logging.getLogger(__name__)

MAIL_ACCEPT = "application/vnd.aconex.mail.v2+xml"
DOC_ACCEPT = "application/xml"
CREATE_MAIL_XML_TYPE = "application/vnd.aconex.mail.v2+xml"


def _local(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _findtext(node: ET.Element, *names: str) -> str:
    wanted = {n.lower() for n in names}
    if _local(node.tag).lower() in wanted:
        return (node.text or "").strip()
    for child in node.iter():
        if _local(child.tag).lower() in wanted:
            return (child.text or "").strip()
    return ""


def _children_named(root: ET.Element, *names: str) -> list[ET.Element]:
    wanted = {n.lower() for n in names}
    return [el for el in root.iter() if _local(el.tag).lower() in wanted]


class AconexCdeClient(CdeClient):
    """Live Oracle Aconex REST client. Construct only when oauth_client_ready()."""

    vendor = "aconex"

    def __init__(
        self,
        *,
        access_token: Optional[str] = None,
        api_base: Optional[str] = None,
        http: Optional[httpx.AsyncClient] = None,
    ) -> None:
        if not config.aconex_enabled():
            raise CdeNotConfiguredError()
        token = (access_token if access_token is not None else config.access_token()).strip()
        self._token = token
        self._api_base = (api_base or config.api_base()).rstrip("/")
        self._http = http

    async def _ensure_token(self) -> str:
        if self._token:
            return self._token
        if not (config.client_id() and config.client_secret()):
            raise CdeNotConfiguredError()
        # User-Bound Integration: client_credentials against the Lobby.
        # https://help.aconex.com/apis/implement-smart-construction-platform-oauth/
        basic = base64.b64encode(
            f"{config.client_id()}:{config.client_secret()}".encode("utf-8")
        ).decode("ascii")
        data: dict[str, str] = {"grant_type": "client_credentials"}
        if config.aconex_user_id():
            data["user_id"] = config.aconex_user_id()
        if config.aconex_user_site():
            data["user_site"] = config.aconex_user_site()
        url = f"{config.lobby_url()}/auth/token"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    headers={
                        "Authorization": f"Basic {basic}",
                        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
                    },
                    data=data,
                )
        except httpx.HTTPError as exc:
            raise CdeError(f"Aconex Lobby token request failed: {exc}") from exc
        if response.status_code >= 400:
            raise CdeNotConfiguredError(
                "Aconex OAuth token request failed "
                f"(HTTP {response.status_code}). Check Lobby client credentials. "
                "The Fork will not invent a local CDE register."
            )
        payload = response.json()
        token = str(payload.get("access_token") or "").strip()
        if not token:
            raise CdeNotConfiguredError(
                "Aconex Lobby returned no access_token. The client is not configured."
            )
        self._token = token
        return token

    async def _request(
        self,
        method: str,
        path: str,
        *,
        accept: str = MAIL_ACCEPT,
        params: Optional[dict[str, Any]] = None,
        content: Optional[bytes] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> httpx.Response:
        token = await self._ensure_token()
        merged = {"Authorization": f"Bearer {token}", "Accept": accept}
        if headers:
            merged.update(headers)
        client = self._http
        owns = False
        if client is None:
            client = httpx.AsyncClient(base_url=self._api_base, timeout=30.0)
            owns = True
        try:
            response = await client.request(
                method,
                path,
                params=params,
                content=content,
                headers=merged,
            )
        except httpx.HTTPError as exc:
            raise CdeError(f"Aconex request {method} {path} failed: {exc}") from exc
        finally:
            if owns:
                await client.aclose()
        if response.status_code >= 400:
            raise CdeError(
                f"Aconex {method} {path} returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        return response

    async def list_mail(
        self, cde_project_id: str, *, mailbox: str = "inbox"
    ) -> list[CdeMail]:
        path = f"/api/projects/{cde_project_id}/mail"
        response = await self._request(
            "GET", path, params={"mail_box": mailbox or "inbox"}
        )
        return [_mail_from_xml(el) for el in _children_named(_parse_xml(response), "Mail")]

    async def get_mail(self, cde_project_id: str, mail_id: str) -> CdeMail:
        path = f"/api/projects/{cde_project_id}/mail/{mail_id}"
        response = await self._request("GET", path)
        root = _parse_xml(response)
        mails = _children_named(root, "Mail")
        node = mails[0] if mails else root
        return _mail_from_xml(node)

    async def list_documents(self, cde_project_id: str) -> list[CdeDocument]:
        path = f"/api/projects/{cde_project_id}/register"
        # return_fields identifiers come from View Document Schema / List Documents.
        params = {
            "search_type": "NUMBER_LIMITED",
            "search_result_size": "50",
            "return_fields": "docno,title,filename,doctype,revision,statusid",
        }
        response = await self._request("GET", path, accept=DOC_ACCEPT, params=params)
        root = _parse_xml(response)
        nodes = _children_named(root, "Document", "SearchResult")
        out: list[CdeDocument] = []
        seen: set[str] = set()
        for node in nodes:
            doc = _document_from_xml(node)
            if doc.id and doc.id not in seen:
                seen.add(doc.id)
                out.append(doc)
        return out

    async def get_document(
        self, cde_project_id: str, document_id: str
    ) -> CdeDocument:
        path = f"/api/projects/{cde_project_id}/register/{document_id}/metadata"
        response = await self._request("GET", path, accept=DOC_ACCEPT)
        root = _parse_xml(response)
        nodes = _children_named(root, "Document")
        node = nodes[0] if nodes else root
        return _document_from_xml(node, fallback_id=document_id)

    async def download_document(
        self, cde_project_id: str, document_id: str
    ) -> bytes:
        path = f"/api/projects/{cde_project_id}/register/{document_id}"
        response = await self._request("GET", path, accept="*/*")
        return response.content

    async def list_rfis(self, cde_project_id: str) -> list[CdeMail]:
        mail = await self.list_mail(cde_project_id)
        return [m for m in mail if _looks_like(m, "rfi", "request for information")]

    async def list_transmittals(self, cde_project_id: str) -> list[CdeMail]:
        mail = await self.list_mail(cde_project_id)
        return [m for m in mail if _looks_like(m, "transmittal")]

    async def post_mail(
        self, cde_project_id: str, draft: CdeMailDraft
    ) -> CdePostResult:
        # Create Mail / Register Mail: POST /api/projects/{projectid}/mail/
        path = f"/api/projects/{cde_project_id}/mail/"
        xml_body = _mail_create_xml(draft)
        boundary = "----theforkCdeBoundary7f3a"
        payload = (
            f"--{boundary}\r\n"
            f"Content-Type: {CREATE_MAIL_XML_TYPE}\r\n"
            "\r\n"
            f"{xml_body}\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")
        response = await self._request(
            "POST",
            path,
            headers={"Content-Type": f"multipart/mixed; boundary={boundary}"},
            content=payload,
        )
        root = _parse_xml(response)
        cde_id = (
            _findtext(root, "MailId", "mailId", "Id")
            or _findtext(root, "DocumentId")
        )
        reference = _findtext(root, "MailNo", "mailNo", "Reference", "ref")
        if not cde_id:
            # Some Create Mail responses return the new id as plain text.
            cde_id = (response.text or "").strip()[:80]
        if not cde_id:
            raise CdeError(
                "Aconex Create Mail succeeded but returned no mail id. "
                "Refusing to allocate a Fork RFI number."
            )
        return CdePostResult(
            cde_id=cde_id,
            reference=reference,
            status="posted",
            vendor=self.vendor,
            raw={"http_status": response.status_code},
        )


def _parse_xml(response: httpx.Response) -> ET.Element:
    text = response.text or ""
    if not text.strip():
        return ET.Element("Empty")
    try:
        return ET.fromstring(text)
    except ET.ParseError:
        wrapper = ET.Element("Raw")
        wrapper.text = text
        return wrapper


def _mail_from_xml(node: ET.Element) -> CdeMail:
    mail_id = _findtext(node, "MailId", "mailId", "Id")
    return CdeMail(
        id=mail_id or _findtext(node, "mailid"),
        subject=_findtext(node, "Subject", "subject"),
        body=_findtext(node, "Body", "body"),
        mail_type=_findtext(node, "MailType", "CorrespondenceType", "corrtype"),
        correspondence_type=_findtext(
            node, "CorrespondenceType", "MailType", "corrtype"
        ),
        status=_findtext(node, "Status", "status"),
        reference=_findtext(node, "MailNo", "mailNo", "Reference"),
        mailbox=_findtext(node, "Mailbox", "mail_box") or "inbox",
    )


def _document_from_xml(node: ET.Element, fallback_id: str = "") -> CdeDocument:
    doc_id = (
        _findtext(node, "DocumentId", "documentId", "documentid")
        or fallback_id
    )
    filename = _findtext(node, "filename", "FileName", "Filename")
    title = _findtext(node, "title", "Title")
    if not filename:
        filename = (title or doc_id or "aconex-document") + ".bin"
    return CdeDocument(
        id=doc_id,
        title=title or filename,
        filename=filename,
        revision=_findtext(node, "revision", "Revision"),
        doc_type=_findtext(node, "doctype", "DocType", "DocumentType"),
        status=_findtext(node, "statusid", "Status", "status"),
    )


def _mail_create_xml(draft: CdeMailDraft) -> str:
    """Minimal Create Mail XML. Field names follow the Mail v2 payload family.

    Recipients / MailTypeId are project-specific (View Mail Creation Schema).
    When Oracle credentials exist, callers should pass mail_type_id from that
    schema. This slice does not invent extra undocumented elements.
    """
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<Mail>",
        f"  <Subject>{xml_escape(draft.subject)}</Subject>",
        f"  <Body>{xml_escape(draft.body)}</Body>",
    ]
    if draft.mail_type_id:
        parts.append(f"  <MailTypeId>{xml_escape(str(draft.mail_type_id))}</MailTypeId>")
    parts.append("</Mail>")
    return "\n".join(parts)


def _looks_like(mail: CdeMail, *needles: str) -> bool:
    blob = f"{mail.mail_type} {mail.correspondence_type} {mail.subject}".lower()
    return any(n in blob for n in needles)

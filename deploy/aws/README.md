# AWS deploy — document bytes

`.github/workflows/deploy-aws.yml` builds one image and rolls the existing
ECS service `the-fork` on cluster `cerebrum` (`us-west-2`). It reuses the
task definition that is already on the service. It does not create a volume,
and it does not change IAM. The live task therefore still has only the
container's own disk until an owner attaches storage.

Uploaded document bytes used to live only under `DATA_DIR` on that disk. A
new task can start without them. This build adds a filesystem byte store
and a startup guard. It does not add an S3 client.

## Do not turn the guard on before the volume exists

`REQUIRE_DURABLE_DOCUMENT_STORE` defaults to unset.

- Unset: production boots, logs that the guard is **not enforcing**, and
  still accepts uploads on `DOCUMENT_BYTE_STORE=local` (the container disk
  when `DATA_DIR` is not a separate mount). This is the current live task.
  Setting the flag before a volume is mounted makes the process exit at
  startup and the service will not become healthy.
- Set to `1` only after `DOCUMENT_BYTE_STORE=efs` and the mount check
  succeeds. The check compares mount identity with the container root
  (device, mount point, filesystem type) and writes, reads, and deletes a
  probe file. A directory that merely exists is not enough.

The web process and the ingest worker run the same guard
(`app.main._validate_startup_env`).

`DOCUMENT_BYTE_STORE=efs` is checked even when the flag is unset. Leave the
backend at `local` until the mount is actually attached. Selecting `efs`
against the container disk refuses to boot.

## Owner steps — EFS

Do these in the AWS console or CLI. The deploy workflow will not do them.

1. Create an EFS file system in `us-west-2`, in the same VPC as the ECS
   service, with a security group that allows NFS (2049) from the task
   security group.
2. Add an EFS volume to the **web** task definition and to the **worker**
   task definition if the worker runs as its own task. Mount it at one
   path, for example `/mnt/efs/documents`. Use an access point if you want
   a fixed uid.
3. Set on both tasks:
   - `DOCUMENT_STORE_ROOT=/mnt/efs/documents`
   - `DOCUMENT_BYTE_STORE=efs`
   - Leave `REQUIRE_DURABLE_DOCUMENT_STORE` unset for the first boot.
4. Deploy this image (or roll the service after the task definition
   revision is current). Confirm the task is healthy and the log contains
   `document byte store: efs mount verified`.
5. Copy any bytes that are still on the old disk onto the mount. See
   `docs/document-bytes-migration.md`. Rows keep their `file_path`.
6. Set `REQUIRE_DURABLE_DOCUMENT_STORE=1` and roll again. From then on a
   task whose mount is missing or is the container filesystem will not
   start, and it will not accept uploads onto ephemeral disk.

The task execution role needs `elasticfilesystem:ClientMount` (and
`ClientWrite` if the access point requires it) on that file system. The
application itself uses the mount as a directory; it does not call the EFS
API.

## Owner steps — S3, if you choose that instead

S3 is not implemented in this build. There is no client, no bucket config,
and no fallback that pretends to store an object. A follow-up has to add a
backend and these IAM actions on the document bucket, on the **task role**
(not only the execution role):

- `s3:PutObject`
- `s3:GetObject`
- `s3:DeleteObject`
- `s3:ListBucket` (for the absence check; a missing key must be
  distinguishable from an access-denied response)

Do not point `DOCUMENT_BYTE_STORE` at `s3`. The process will refuse to
start. Keep `local` or `efs` until that backend exists.

Until then, EFS (or another mounted volume) is the durable store this
image can use.

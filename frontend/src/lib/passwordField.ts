// The one rule for a password field's autocomplete hint.
//
// The values are the HTML autofill field names (WHATWG HTML, "autofill"):
// a field that asks for the secret the user already has is
// `current-password`; a field where the user chooses a secret is
// `new-password`. The right hint lets the browser's password manager fill the
// sign-in form and offer to save (or generate) a new password, so the app
// itself never has to keep one.

export type PasswordPurpose = 'sign-in' | 'new'

export const PASSWORD_AUTOCOMPLETE = {
  'sign-in': 'current-password',
  'new': 'new-password',
} as const satisfies Record<PasswordPurpose, string>

export function passwordAutoComplete(purpose: PasswordPurpose): string {
  return PASSWORD_AUTOCOMPLETE[purpose]
}

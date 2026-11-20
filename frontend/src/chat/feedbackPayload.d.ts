export type FeedbackKind = 'correct' | 'wrong_figure' | 'wrong_document' | 'missing'
export declare const FEEDBACK_KINDS: { kind: FeedbackKind; label: string }[]
export declare const NOTE_MAX: number
export declare const UNIT_MAX: number
export declare const HEAD_CHARS: number
export declare function collapsed(text: string | null | undefined): string
export declare function answerHead(content: string | null | undefined): string
export interface FeedbackInput {
  projectId: string
  conversationId: string
  assistantIndex: number
  kind: FeedbackKind
  value?: string
  unit?: string
  note?: string
  content: string
}
export type FeedbackBody = Record<string, string | number>
export declare function buildFeedbackBody(input: FeedbackInput): { body: FeedbackBody; error?: undefined } | { error: string; body?: undefined }
export interface FeedbackEcho {
  status: string
  message_id: string
  answer_head: string
}
export declare function echoMatches(response: FeedbackEcho | null | undefined, content: string): boolean

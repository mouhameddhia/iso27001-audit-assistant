import type {
  AuditFinding,
  AuditSession,
  ImportedDocument,
  ReviewedFinding,
  Role,
  SessionSummary,
  User,
} from './types'

const TOKEN_KEY = 'iso27001_audit_token'

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY)
  } catch {
    return null
  }
}

export function setToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token)
    else localStorage.removeItem(TOKEN_KEY)
  } catch {
    // localStorage unavailable (private browsing, blocked storage) -- the session just won't
    // persist across a reload; not fatal.
  }
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getToken()
  const headers = new Headers(options.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (options.body && !(options.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }

  const response = await fetch(`/api${path}`, { ...options, headers })
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = await response.json()
      detail = body.detail ?? detail
    } catch {
      // body wasn't JSON -- keep the status text
    }
    throw new ApiError(response.status, detail)
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

function json(body: unknown): RequestInit {
  return { method: 'POST', body: JSON.stringify(body) }
}

export const api = {
  login: (username: string, password: string) =>
    request<{ access_token: string; username: string; role: Role; full_name: string }>(
      '/auth/login',
      json({ username, password }),
    ),
  me: () => request<User>('/auth/me'),

  listUsers: () => request<User[]>('/users'),
  createUser: (body: { username: string; password: string; full_name: string; role: Role }) =>
    request<User>('/users', json(body)),

  listSessions: () => request<SessionSummary[]>('/sessions'),
  createSession: (body: {
    title: string
    client_name: string
    client_aliases: string[]
    scope: string
    standards: string[]
    audit_team: { name: string; role: string }[]
    reference?: string
    start_date?: string
    end_date?: string
  }) => request<SessionSummary>('/sessions', json(body)),
  getSession: (sessionId: string) => request<AuditSession>(`/sessions/${sessionId}`),
  closeSession: (sessionId: string) => request<SessionSummary>(`/sessions/${sessionId}/close`, { method: 'POST' }),
  setCertificationDecision: (sessionId: string, recommends: boolean, decidedBy?: string) =>
    request<AuditSession>(
      `/sessions/${sessionId}/certification-decision`,
      json({ recommends, decided_by: decidedBy }),
    ),

  addFinding: (sessionId: string, observation: string, language = 'fr') =>
    request<AuditFinding>(`/sessions/${sessionId}/findings`, json({ observation, language })),
  reviewFinding: (
    sessionId: string,
    index: number,
    body: { approve: boolean; severity?: string | null; edited_text?: string | null; reviewer?: string | null },
  ) => request<ReviewedFinding>(`/sessions/${sessionId}/findings/${index}/review`, json(body)),

  listDocuments: (sessionId: string) => request<ImportedDocument[]>(`/sessions/${sessionId}/documents`),
  importDocument: (sessionId: string, file: File, docType: string) => {
    const form = new FormData()
    form.append('file', file)
    form.append('doc_type', docType)
    return request<ImportedDocument>(`/sessions/${sessionId}/documents`, { method: 'POST', body: form })
  },
  getSidecar: (sessionId: string, docId: string) =>
    request<{ doc_id: string; status: string; content: string }>(
      `/sessions/${sessionId}/documents/${docId}/sidecar`,
    ),
  confirmDocument: (sessionId: string, docId: string) =>
    request<ImportedDocument>(`/sessions/${sessionId}/documents/${docId}/confirm`, { method: 'POST' }),

  downloadReport: async (sessionId: string, format: 'docx' | 'pdf'): Promise<Blob> => {
    const token = getToken()
    const headers = new Headers()
    if (token) headers.set('Authorization', `Bearer ${token}`)
    const response = await fetch(`/api/sessions/${sessionId}/report?format=${format}`, { headers })
    if (!response.ok) throw new ApiError(response.status, response.statusText)
    return response.blob()
  },
}

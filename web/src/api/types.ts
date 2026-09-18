// Mirrors the API's Pydantic response models (api/schemas.py, src/generation/models.py,
// src/report/session.py). Kept in one place so a backend field rename is a one-file fix here.

export type Role = 'auditeur' | 'chef_equipe' | 'administrateur'

export interface User {
  id: string
  username: string
  full_name: string
  role: Role
  is_active: boolean
}

export interface TeamMember {
  name: string
  role: string
}

export interface SessionMetadata {
  session_id: string
  title: string
  client_name: string
  client_aliases: string[]
  scope: string
  standards: string[]
  audit_team: TeamMember[]
  reference: string | null
  start_date: string | null
  end_date: string | null
}

export interface SupportingSource {
  chunk_id: string
  doc_id: string
  doc_title: string
  references: string[]
  rerank_score: number
}

export type FindingType = 'constat' | 'non_conformite' | 'observation' | 'opportunite_amelioration'
export type EvidenceStatus = 'supported' | 'partial' | 'insufficient'

export interface AuditFinding {
  observation: string
  finding: string
  finding_type: FindingType | null
  requirement: string | null
  iso_reference: string[]
  justification: string | null
  risk: string | null
  recommendation: string | null
  evidence_status: EvidenceStatus
  requires_human_review: boolean
  confidence: number
  supporting_sources: SupportingSource[]
  document_sources: SupportingSource[]
  validation_notes: string[]
}

export type ReviewDecision = 'pending' | 'approved' | 'rejected'
export type Severity = 'majeure' | 'mineure'

export interface ReviewedFinding {
  finding: AuditFinding
  decision: ReviewDecision
  severity: Severity | null
  edited_text: string | null
  reviewer: string | null
  reviewed_at: string | null
}

export type ImportStatus = 'pending_review' | 'confirmed'

export interface ImportedDocument {
  doc_id: string
  filename: string
  doc_type: string
  sidecar_path: string
  status: ImportStatus
  chunk_count: number
  imported_at: string | null
  confirmed_at: string | null
}

export type CertificationDecision = 'recommande' | 'ne_recommande_pas'

export interface AuditSession {
  metadata: SessionMetadata
  findings: ReviewedFinding[]
  imported_documents: ImportedDocument[]
  certification_decision: CertificationDecision | null
  certification_decided_by: string | null
  certification_decided_at: string | null
}

export interface SessionSummary {
  id: string
  title: string
  client_name: string
  reference: string | null
  status: 'open' | 'closed'
  owner_id: string
  pending_count: number
  approved_count: number
  rejected_count: number
  imported_documents: number
}

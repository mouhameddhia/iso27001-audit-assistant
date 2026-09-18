import { useEffect, useState, type FormEvent } from 'react'
import { useParams } from 'react-router-dom'
import { api, ApiError } from '../api/client'
import type { AuditSession, ImportedDocument, ReviewedFinding } from '../api/types'
import { useAuth } from '../auth/AuthContext'

const FINDING_TYPE_LABELS: Record<string, string> = {
  constat: 'Constat',
  non_conformite: 'Non-conformité',
  observation: 'Observation',
  opportunite_amelioration: "Opportunité d'amélioration",
}

function DecisionBadge({ decision }: { decision: ReviewedFinding['decision'] }) {
  if (decision === 'approved') return <span className="badge badge-approved">Approuvé</span>
  if (decision === 'rejected') return <span className="badge badge-rejected">Rejeté</span>
  return <span className="badge badge-pending">En attente</span>
}

function FindingCard({
  reviewed,
  index,
  sessionId,
  onReviewed,
}: {
  reviewed: ReviewedFinding
  index: number
  sessionId: string
  onReviewed: () => void
}) {
  const { finding } = reviewed
  const [editing, setEditing] = useState(false)
  const [editedText, setEditedText] = useState(reviewed.edited_text ?? finding.finding)
  const [severity, setSeverity] = useState(reviewed.severity ?? 'mineure')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const needsSeverity = finding.finding_type === 'non_conformite'

  async function review(approve: boolean) {
    setBusy(true)
    setError(null)
    try {
      await api.reviewFinding(sessionId, index, {
        approve,
        severity: needsSeverity ? severity : undefined,
        edited_text: editing ? editedText : undefined,
        reviewer: undefined,
      })
      setEditing(false)
      onReviewed()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Échec de la revue.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card stack">
      <div className="row-between">
        <div className="row">
          <strong>{FINDING_TYPE_LABELS[finding.finding_type ?? ''] ?? 'Constat'}</strong>
          <DecisionBadge decision={reviewed.decision} />
          {reviewed.severity && <span className="badge badge-neutral">{reviewed.severity}</span>}
          {finding.requires_human_review && reviewed.decision === 'pending' && (
            <span className="badge badge-pending">Revue requise</span>
          )}
        </div>
        <span className="muted">Confiance : {(finding.confidence * 100).toFixed(0)}%</span>
      </div>

      <p className="muted" style={{ margin: 0 }}>
        Observation : {finding.observation}
      </p>

      {editing ? (
        <textarea value={editedText} onChange={(e) => setEditedText(e.target.value)} />
      ) : (
        <p style={{ margin: 0 }}>{reviewed.edited_text ?? finding.finding}</p>
      )}

      {finding.risk && (
        <p className="muted" style={{ margin: 0 }}>
          <strong>Risque :</strong> {finding.risk}
        </p>
      )}
      {finding.recommendation && (
        <p className="muted" style={{ margin: 0 }}>
          <strong>Recommandation :</strong> {finding.recommendation}
        </p>
      )}
      {finding.iso_reference.length > 0 && (
        <p className="muted" style={{ margin: 0 }}>
          <strong>Référence(s) ISO :</strong> {finding.iso_reference.join(', ')}
        </p>
      )}
      {finding.document_sources.length > 0 && (
        <div className="alert alert-info">Contexte documentaire client utilisé — revue humaine obligatoire.</div>
      )}
      {finding.evidence_status === 'insufficient' && (
        <div className="alert alert-warning">Preuves insuffisantes pour confirmer ce constat.</div>
      )}

      {error && <div className="alert alert-error">{error}</div>}

      {reviewed.decision === 'pending' && (
        <div className="row">
          {needsSeverity && (
            <select
              value={severity}
              onChange={(e) => setSeverity(e.target.value as 'majeure' | 'mineure')}
              style={{ width: 'auto' }}
            >
              <option value="mineure">Mineure</option>
              <option value="majeure">Majeure</option>
            </select>
          )}
          <button className="btn btn-sm" onClick={() => setEditing((v) => !v)}>
            {editing ? 'Annuler la modification' : 'Modifier le texte'}
          </button>
          <button className="btn btn-primary btn-sm" disabled={busy} onClick={() => review(true)}>
            Approuver
          </button>
          <button className="btn btn-danger btn-sm" disabled={busy} onClick={() => review(false)}>
            Rejeter
          </button>
        </div>
      )}
    </div>
  )
}

function FindingsTab({ session, sessionId, refresh }: { session: AuditSession; sessionId: string; refresh: () => void }) {
  const [observation, setObservation] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setError(null)
    setSubmitting(true)
    try {
      await api.addFinding(sessionId, observation)
      setObservation('')
      refresh()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Échec de la génération du constat.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="stack">
      <form className="card stack" onSubmit={handleSubmit}>
        <label htmlFor="observation">Observation de l'auditeur</label>
        <textarea
          id="observation"
          value={observation}
          onChange={(e) => setObservation(e.target.value)}
          placeholder="Ex. Le contrôle des accès privilégiés n'est pas revu périodiquement."
          required
        />
        {error && <div className="alert alert-error">{error}</div>}
        <button className="btn btn-primary" type="submit" disabled={submitting}>
          {submitting ? 'Génération en cours…' : 'Générer le constat'}
        </button>
      </form>

      {session.findings.length === 0 && <p className="muted">Aucun constat pour le moment.</p>}
      {[...session.findings]
        .map((reviewed, index) => ({ reviewed, index }))
        .reverse()
        .map(({ reviewed, index }) => (
          <FindingCard key={index} reviewed={reviewed} index={index} sessionId={sessionId} onReviewed={refresh} />
        ))}
    </div>
  )
}

function DocumentRow({
  doc,
  sessionId,
  onConfirmed,
}: {
  doc: ImportedDocument
  sessionId: string
  onConfirmed: () => void
}) {
  const [sidecar, setSidecar] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function viewSidecar() {
    setError(null)
    try {
      const result = await api.getSidecar(sessionId, doc.doc_id)
      setSidecar(result.content)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Échec du chargement.')
    }
  }

  async function confirm() {
    setBusy(true)
    setError(null)
    try {
      await api.confirmDocument(sessionId, doc.doc_id)
      onConfirmed()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Échec de la confirmation de l'import.")
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card stack">
      <div className="row-between">
        <div>
          <strong>{doc.filename}</strong> <span className="muted">({doc.doc_type})</span>
        </div>
        {doc.status === 'confirmed' ? (
          <span className="badge badge-approved">Confirmé · {doc.chunk_count} passage(s)</span>
        ) : (
          <span className="badge badge-pending">En attente de revue</span>
        )}
      </div>
      {error && <div className="alert alert-error">{error}</div>}
      {doc.status === 'pending_review' && (
        <div className="stack">
          <p className="muted" style={{ margin: 0 }}>
            Avant de confirmer, relisez le contenu anonymisé ci-dessous : aucune donnée réelle du
            client ne doit y apparaître.
          </p>
          <div className="row">
            <button className="btn btn-sm" onClick={viewSidecar}>
              Voir le contenu anonymisé
            </button>
            <button className="btn btn-primary btn-sm" disabled={busy} onClick={confirm}>
              Confirmer l'import
            </button>
          </div>
          {sidecar !== null && <pre className="sidecar">{sidecar}</pre>}
        </div>
      )}
    </div>
  )
}

function DocumentsTab({ session, sessionId, refresh }: { session: AuditSession; sessionId: string; refresh: () => void }) {
  const [file, setFile] = useState<File | null>(null)
  const [docType, setDocType] = useState('previous_report')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!file) return
    setError(null)
    setSubmitting(true)
    try {
      await api.importDocument(sessionId, file, docType)
      setFile(null)
      refresh()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Échec de l'import du document.")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="stack">
      <form className="card stack" onSubmit={handleSubmit}>
        <p className="muted" style={{ margin: 0 }}>
          PDF, DOCX, XLSX ou CSV. Le document est anonymisé avant tout traitement ; vous devrez
          relire et confirmer son contenu anonymisé avant qu'il ne soit exploité.
        </p>
        <div className="row">
          <select value={docType} onChange={(e) => setDocType(e.target.value)} style={{ width: 'auto' }}>
            <option value="previous_report">Rapport d'audit précédent</option>
            <option value="soa">Déclaration d'applicabilité</option>
            <option value="risk_analysis">Analyse de risques</option>
            <option value="procedure">Procédure</option>
            <option value="policy">Politique SSI</option>
          </select>
          <input
            type="file"
            accept=".pdf,.docx,.xlsx,.csv"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            required
          />
          <button className="btn btn-primary" type="submit" disabled={submitting || !file}>
            {submitting ? 'Import…' : 'Importer'}
          </button>
        </div>
        {error && <div className="alert alert-error">{error}</div>}
      </form>

      {session.imported_documents.length === 0 && <p className="muted">Aucun document importé.</p>}
      {session.imported_documents.map((doc) => (
        <DocumentRow key={doc.doc_id} doc={doc} sessionId={sessionId} onConfirmed={refresh} />
      ))}
    </div>
  )
}

function CertificationDecisionCard({
  session,
  sessionId,
  refresh,
}: {
  session: AuditSession
  sessionId: string
  refresh: () => void
}) {
  const { user } = useAuth()
  const canDecide = user?.role === 'chef_equipe' || user?.role === 'administrateur'
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function decide(recommends: boolean) {
    setBusy(true)
    setError(null)
    try {
      await api.setCertificationDecision(sessionId, recommends, user?.full_name)
      refresh()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Échec de l’enregistrement de la décision.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card stack">
      <strong>Décision de certification</strong>
      <p className="muted" style={{ margin: 0 }}>
        Cette décision n'est jamais proposée par l'IA : elle doit être enregistrée explicitement par
        l'Auditeur Principal avant transmission du rapport au client.
      </p>
      {session.certification_decision === 'recommande' && (
        <div className="alert alert-info">
          Recommande le maintien de la certification — enregistré par {session.certification_decided_by ?? '—'}.
        </div>
      )}
      {session.certification_decision === 'ne_recommande_pas' && (
        <div className="alert alert-warning">
          Ne recommande pas le maintien de la certification — enregistré par{' '}
          {session.certification_decided_by ?? '—'}.
        </div>
      )}
      {session.certification_decision === null && (
        <div className="alert alert-warning">Décision non enregistrée — le rapport l'indiquera comme « à confirmer ».</div>
      )}
      {error && <div className="alert alert-error">{error}</div>}
      {canDecide ? (
        <div className="row">
          <button className="btn btn-primary btn-sm" disabled={busy} onClick={() => decide(true)}>
            Recommander la certification
          </button>
          <button className="btn btn-danger btn-sm" disabled={busy} onClick={() => decide(false)}>
            Ne pas recommander
          </button>
        </div>
      ) : (
        <p className="muted" style={{ margin: 0 }}>
          Seul un chef d'équipe ou un administrateur peut enregistrer cette décision.
        </p>
      )}
    </div>
  )
}

function ReportTab({
  session,
  sessionId,
  refresh,
}: {
  session: AuditSession
  sessionId: string
  refresh: () => void
}) {
  const [error, setError] = useState<string | null>(null)
  const [downloading, setDownloading] = useState<'docx' | 'pdf' | null>(null)

  const approvedCount = session.findings.filter((f) => f.decision === 'approved').length

  async function download(format: 'docx' | 'pdf') {
    setError(null)
    setDownloading(format)
    try {
      const blob = await api.downloadReport(sessionId, format)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `rapport_${session.metadata.reference ?? sessionId}.${format}`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Échec de la génération du rapport.')
    } finally {
      setDownloading(null)
    }
  }

  return (
    <div className="stack">
      <CertificationDecisionCard session={session} sessionId={sessionId} refresh={refresh} />
      <div className="card stack">
        <p style={{ margin: 0 }}>
          {approvedCount} constat(s) approuvé(s) prêt(s) à figurer dans le rapport.
        </p>
        {approvedCount === 0 && (
          <div className="alert alert-warning">
            Aucun constat approuvé : approuvez au moins un constat dans l'onglet « Constats » avant
            de générer le rapport.
          </div>
        )}
        {error && <div className="alert alert-error">{error}</div>}
        <div className="row">
          <button className="btn btn-primary" disabled={downloading !== null} onClick={() => download('docx')}>
            {downloading === 'docx' ? 'Génération…' : 'Télécharger (Word)'}
          </button>
          <button className="btn" disabled={downloading !== null} onClick={() => download('pdf')}>
            {downloading === 'pdf' ? 'Génération…' : 'Télécharger (PDF)'}
          </button>
        </div>
      </div>
    </div>
  )
}

export function SessionDetailPage() {
  const { sessionId } = useParams<{ sessionId: string }>()
  const [session, setSession] = useState<AuditSession | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [tab, setTab] = useState<'findings' | 'documents' | 'report'>('findings')

  function refresh() {
    if (!sessionId) return
    api
      .getSession(sessionId)
      .then(setSession)
      .catch((err) => setError(err instanceof ApiError ? err.message : 'Échec du chargement de la mission.'))
  }

  useEffect(refresh, [sessionId])

  if (error) return <div className="alert alert-error">{error}</div>
  if (!session) return <p className="muted">Chargement…</p>

  return (
    <div className="stack">
      <div>
        <h2 style={{ margin: 0 }}>{session.metadata.title}</h2>
        <p className="muted" style={{ margin: 0 }}>
          {session.metadata.client_name} · {session.metadata.reference ?? 'sans référence'}
        </p>
      </div>

      <div className="tabs">
        <div className={`tab ${tab === 'findings' ? 'active' : ''}`} onClick={() => setTab('findings')}>
          Constats
        </div>
        <div className={`tab ${tab === 'documents' ? 'active' : ''}`} onClick={() => setTab('documents')}>
          Documents
        </div>
        <div className={`tab ${tab === 'report' ? 'active' : ''}`} onClick={() => setTab('report')}>
          Rapport
        </div>
      </div>

      {tab === 'findings' && <FindingsTab session={session} sessionId={session.metadata.session_id} refresh={refresh} />}
      {tab === 'documents' && <DocumentsTab session={session} sessionId={session.metadata.session_id} refresh={refresh} />}
      {tab === 'report' && <ReportTab session={session} sessionId={session.metadata.session_id} refresh={refresh} />}
    </div>
  )
}

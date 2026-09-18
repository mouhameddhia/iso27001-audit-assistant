import { useEffect, useState, type FormEvent } from 'react'
import { Link } from 'react-router-dom'
import { api, ApiError } from '../api/client'
import type { SessionSummary } from '../api/types'

function StatusBadge({ session }: { session: SessionSummary }) {
  if (session.status === 'closed') return <span className="badge badge-neutral">Clôturée</span>
  if (session.pending_count > 0) return <span className="badge badge-pending">{session.pending_count} en attente</span>
  return <span className="badge badge-approved">À jour</span>
}

function NewSessionForm({ onCreated }: { onCreated: (s: SessionSummary) => void }) {
  const [open, setOpen] = useState(false)
  const [title, setTitle] = useState('')
  const [clientName, setClientName] = useState('')
  const [clientAliases, setClientAliases] = useState('')
  const [scope, setScope] = useState('')
  const [standards, setStandards] = useState('ISO/IEC 27001:2022')
  const [reference, setReference] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setError(null)
    setSubmitting(true)
    try {
      const session = await api.createSession({
        title,
        client_name: clientName,
        client_aliases: clientAliases.split(',').map((s) => s.trim()).filter(Boolean),
        scope,
        standards: standards.split(',').map((s) => s.trim()).filter(Boolean),
        audit_team: [],
        reference: reference || undefined,
      })
      onCreated(session)
      setOpen(false)
      setTitle('')
      setClientName('')
      setClientAliases('')
      setScope('')
      setReference('')
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Échec de la création de la session.')
    } finally {
      setSubmitting(false)
    }
  }

  if (!open) {
    return (
      <button className="btn btn-primary" onClick={() => setOpen(true)}>
        + Nouvelle mission
      </button>
    )
  }

  return (
    <form className="card stack" onSubmit={handleSubmit}>
      <div className="row-between">
        <h3 style={{ margin: 0 }}>Nouvelle mission d'audit</h3>
        <button type="button" className="btn btn-sm" onClick={() => setOpen(false)}>
          Annuler
        </button>
      </div>
      {error && <div className="alert alert-error">{error}</div>}
      <div className="field">
        <label htmlFor="title">Titre du rapport</label>
        <input id="title" value={title} onChange={(e) => setTitle(e.target.value)} required />
      </div>
      <div className="row">
        <div className="field" style={{ flex: 1 }}>
          <label htmlFor="client">Client</label>
          <input id="client" value={clientName} onChange={(e) => setClientName(e.target.value)} required />
        </div>
        <div className="field" style={{ flex: 1 }}>
          <label htmlFor="aliases">Autres noms du client (séparés par des virgules)</label>
          <input id="aliases" value={clientAliases} onChange={(e) => setClientAliases(e.target.value)} />
        </div>
      </div>
      <div className="field">
        <label htmlFor="scope">Périmètre</label>
        <textarea id="scope" value={scope} onChange={(e) => setScope(e.target.value)} required />
      </div>
      <div className="row">
        <div className="field" style={{ flex: 1 }}>
          <label htmlFor="standards">Référentiel(s) (séparés par des virgules)</label>
          <input id="standards" value={standards} onChange={(e) => setStandards(e.target.value)} />
        </div>
        <div className="field" style={{ flex: 1 }}>
          <label htmlFor="reference">Référence de mission</label>
          <input id="reference" value={reference} onChange={(e) => setReference(e.target.value)} />
        </div>
      </div>
      <button className="btn btn-primary" type="submit" disabled={submitting}>
        {submitting ? 'Création…' : 'Créer la mission'}
      </button>
    </form>
  )
}

export function SessionListPage() {
  const [sessions, setSessions] = useState<SessionSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api
      .listSessions()
      .then(setSessions)
      .catch((err) => setError(err instanceof ApiError ? err.message : 'Échec du chargement des missions.'))
  }, [])

  return (
    <div className="stack">
      <div className="row-between">
        <h2 style={{ margin: 0 }}>Missions d'audit</h2>
      </div>
      <NewSessionForm onCreated={(s) => setSessions((prev) => [s, ...(prev ?? [])])} />
      {error && <div className="alert alert-error">{error}</div>}
      {sessions === null && !error && <p className="muted">Chargement…</p>}
      {sessions !== null && sessions.length === 0 && <p className="muted">Aucune mission pour le moment.</p>}
      {sessions !== null && sessions.length > 0 && (
        <div className="card" style={{ padding: 0 }}>
          <table>
            <thead>
              <tr>
                <th>Mission</th>
                <th>Client</th>
                <th>Référence</th>
                <th>Constats</th>
                <th>Documents</th>
                <th>Statut</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map((s) => (
                <tr key={s.id}>
                  <td>
                    <Link to={`/sessions/${s.id}`}>{s.title}</Link>
                  </td>
                  <td>{s.client_name}</td>
                  <td>{s.reference ?? '—'}</td>
                  <td>
                    {s.approved_count} approuvé(s) / {s.rejected_count} rejeté(s)
                  </td>
                  <td>{s.imported_documents}</td>
                  <td>
                    <StatusBadge session={s} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

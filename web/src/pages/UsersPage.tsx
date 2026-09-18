import { useEffect, useState, type FormEvent } from 'react'
import { api, ApiError } from '../api/client'
import type { Role, User } from '../api/types'
import { useAuth } from '../auth/AuthContext'

export function UsersPage() {
  const { user: me } = useAuth()
  const [users, setUsers] = useState<User[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [fullName, setFullName] = useState('')
  const [role, setRole] = useState<Role>('auditeur')
  const [submitting, setSubmitting] = useState(false)
  const [formError, setFormError] = useState<string | null>(null)

  function refresh() {
    api
      .listUsers()
      .then(setUsers)
      .catch((err) => setError(err instanceof ApiError ? err.message : 'Échec du chargement.'))
  }

  useEffect(refresh, [])

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setFormError(null)
    setSubmitting(true)
    try {
      await api.createUser({ username, password, full_name: fullName, role })
      setUsername('')
      setPassword('')
      setFullName('')
      setRole('auditeur')
      refresh()
    } catch (err) {
      setFormError(err instanceof ApiError ? err.message : "Échec de la création de l'utilisateur.")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="stack">
      <h2 style={{ margin: 0 }}>Utilisateurs</h2>

      {me?.role === 'administrateur' && (
        <form className="card stack" onSubmit={handleSubmit}>
          <h3 style={{ margin: 0 }}>Nouvel utilisateur</h3>
          {formError && <div className="alert alert-error">{formError}</div>}
          <div className="row">
            <div className="field" style={{ flex: 1 }}>
              <label htmlFor="new-username">Nom d'utilisateur</label>
              <input id="new-username" value={username} onChange={(e) => setUsername(e.target.value)} required />
            </div>
            <div className="field" style={{ flex: 1 }}>
              <label htmlFor="new-fullname">Nom complet</label>
              <input id="new-fullname" value={fullName} onChange={(e) => setFullName(e.target.value)} required />
            </div>
          </div>
          <div className="row">
            <div className="field" style={{ flex: 1 }}>
              <label htmlFor="new-password">Mot de passe temporaire</label>
              <input
                id="new-password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            <div className="field" style={{ flex: 1 }}>
              <label htmlFor="new-role">Rôle</label>
              <select id="new-role" value={role} onChange={(e) => setRole(e.target.value as Role)}>
                <option value="auditeur">Auditeur</option>
                <option value="chef_equipe">Chef d'équipe</option>
                <option value="administrateur">Administrateur</option>
              </select>
            </div>
          </div>
          <button className="btn btn-primary" type="submit" disabled={submitting}>
            {submitting ? 'Création…' : "Créer l'utilisateur"}
          </button>
        </form>
      )}

      {error && <div className="alert alert-error">{error}</div>}
      {users && (
        <div className="card" style={{ padding: 0 }}>
          <table>
            <thead>
              <tr>
                <th>Nom d'utilisateur</th>
                <th>Nom complet</th>
                <th>Rôle</th>
                <th>Statut</th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.id}>
                  <td>{u.username}</td>
                  <td>{u.full_name}</td>
                  <td>{u.role}</td>
                  <td>{u.is_active ? 'Actif' : 'Inactif'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

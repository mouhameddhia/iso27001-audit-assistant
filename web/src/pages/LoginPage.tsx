import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'
import { ApiError } from '../api/client'

export function LoginPage() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setError(null)
    setSubmitting(true)
    try {
      await login(username, password)
      navigate('/')
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Impossible de se connecter au serveur.")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="login-shell">
      <form className="card login-card stack" onSubmit={handleSubmit}>
        <div>
          <h2 style={{ margin: 0 }}>Connexion</h2>
          <p className="muted">Assistant d'audit ISO/IEC 27001</p>
        </div>
        {error && <div className="alert alert-error">{error}</div>}
        <div className="field">
          <label htmlFor="username">Nom d'utilisateur</label>
          <input id="username" type="text" value={username} onChange={(e) => setUsername(e.target.value)} required autoFocus />
        </div>
        <div className="field">
          <label htmlFor="password">Mot de passe</label>
          <input id="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
        </div>
        <button className="btn btn-primary" type="submit" disabled={submitting}>
          {submitting ? 'Connexion…' : 'Se connecter'}
        </button>
      </form>
    </div>
  )
}

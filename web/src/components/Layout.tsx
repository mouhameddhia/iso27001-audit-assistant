import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'

export function Layout({ children }: { children: React.ReactNode }) {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  return (
    <div className="app-shell">
      <header className="app-header">
        <Link to="/" className="brand">
          Assistant d'audit ISO/IEC 27001
        </Link>
        <div className="row">
          {user && (user.role === 'administrateur' || user.role === 'chef_equipe') && (
            <Link to="/users" className="muted">
              Utilisateurs
            </Link>
          )}
          {user && (
            <span className="muted">
              {user.full_name} · {user.role}
            </span>
          )}
          {user && (
            <button
              className="btn btn-sm"
              onClick={() => {
                logout()
                navigate('/login')
              }}
            >
              Déconnexion
            </button>
          )}
        </div>
      </header>
      <main className="app-main">{children}</main>
    </div>
  )
}

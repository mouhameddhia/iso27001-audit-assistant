import { Navigate, Route, BrowserRouter, Routes } from 'react-router-dom'
import { AuthProvider, useAuth } from './auth/AuthContext'
import { Layout } from './components/Layout'
import { LoginPage } from './pages/LoginPage'
import { SessionDetailPage } from './pages/SessionDetailPage'
import { SessionListPage } from './pages/SessionListPage'
import { UsersPage } from './pages/UsersPage'

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth()
  if (loading) return <p className="muted" style={{ padding: '1.5rem' }}>Chargement…</p>
  if (!user) return <Navigate to="/login" replace />
  return <Layout>{children}</Layout>
}

function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <SessionListPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/sessions/:sessionId"
            element={
              <ProtectedRoute>
                <SessionDetailPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/users"
            element={
              <ProtectedRoute>
                <UsersPage />
              </ProtectedRoute>
            }
          />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  )
}

export default App

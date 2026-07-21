import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "@/components/Layout";
import { Home } from "@/pages/Home";
import { Benchmark } from "@/pages/Benchmark";
import { Leaderboard } from "@/pages/Leaderboard";
import { Login } from "@/pages/Login";
import { Submit } from "@/pages/Submit";
import { Submissions } from "@/pages/Submissions";
import { Profile } from "@/pages/Profile";
import { Admin } from "@/pages/Admin";
import { Privacy } from "@/pages/Privacy";
import { NotFound } from "@/pages/NotFound";
import { AppErrorBoundary } from "@/components/AppErrorBoundary";

export default function App() {
  return (
    <AppErrorBoundary>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Home />} />
            <Route path="/benchmarks/:slug" element={<Benchmark />} />
            <Route path="/leaderboard" element={<Leaderboard />} />
            <Route path="/login" element={<Login />} />
            <Route path="/signup" element={<Navigate to="/login?mode=register" replace />} />
            <Route path="/submit" element={<Submit />} />
            <Route path="/submissions" element={<Submissions />} />
            <Route path="/profile" element={<Profile />} />
            <Route path="/admin" element={<Admin />} />
            <Route path="/privacy" element={<Privacy />} />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AppErrorBoundary>
  );
}

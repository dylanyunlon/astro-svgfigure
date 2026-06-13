import type { APIRoute } from 'astro'

const BACKEND = import.meta.env.BACKEND_URL || process.env.BACKEND_URL || 'http://127.0.0.1:8000'

export const POST: APIRoute = async ({ request }) => {
  const body = await request.json()
  const res = await fetch(`${BACKEND}/api/cell/publish`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const data = await res.json()
  return new Response(JSON.stringify(data), {
    status: res.status,
    headers: { 'Content-Type': 'application/json' },
  })
}

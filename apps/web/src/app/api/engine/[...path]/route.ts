import type { NextRequest } from "next/server";

const ENGINE_API_URL = process.env.ENGINE_API_URL ?? "http://127.0.0.1:8100";

type RouteContext = {
  params: Promise<{ path: string[] }>;
};

async function proxy(request: NextRequest, context: RouteContext) {
  const { path } = await context.params;
  const target = new URL(path.join("/"), `${ENGINE_API_URL}/`);
  target.search = request.nextUrl.search;
  const upstream = await fetch(target, {
    method: request.method,
    headers:
      request.method === "GET"
        ? undefined
        : {
            "Content-Type":
              request.headers.get("content-type") ?? "application/json",
          },
    body:
      request.method === "GET" || request.method === "HEAD"
        ? undefined
        : await request.arrayBuffer(),
    cache: "no-store",
  });
  const headers = new Headers();
  const contentType = upstream.headers.get("content-type");
  if (contentType) headers.set("Content-Type", contentType);
  headers.set("Cache-Control", "no-store");
  if (contentType?.startsWith("text/event-stream")) {
    headers.set("X-Accel-Buffering", "no");
  }
  return new Response(upstream.body, {
    status: upstream.status,
    headers,
  });
}

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest, context: RouteContext) {
  return proxy(request, context);
}

export async function POST(request: NextRequest, context: RouteContext) {
  return proxy(request, context);
}

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND_URL = process.env.TRIBE_API_URL ?? "http://127.0.0.1:8002/analyze";
const MAX_UPLOAD_BYTES = Number(process.env.TRIBE_MAX_UPLOAD_BYTES ?? 500 * 1024 * 1024);
const REQUEST_TIMEOUT_MS = Number(process.env.TRIBE_REQUEST_TIMEOUT_MS ?? 60_000);

export async function POST(request) {
  try {
    const contentLength = Number(request.headers.get("content-length") ?? 0);
    if (Number.isFinite(contentLength) && contentLength > MAX_UPLOAD_BYTES) {
      return Response.json(
        {
          error: `File too large. Maximum is ${Math.floor(MAX_UPLOAD_BYTES / 1024 / 1024)} MB.`,
          code: "FILE_TOO_LARGE",
          maxSize: MAX_UPLOAD_BYTES,
        },
        { status: 413 }
      );
    }

    const contentType = request.headers.get("content-type") ?? "";
    if (!contentType.startsWith("multipart/form-data")) {
      return Response.json(
        {
          error: "Expected multipart upload.",
          code: "INVALID_CONTENT_TYPE",
          details: "Submit the upload as multipart/form-data.",
        },
        { status: 400 }
      );
    }

    const upstreamHeaders = new Headers();
    upstreamHeaders.set("Content-Type", contentType);
    if (Number.isFinite(contentLength) && contentLength > 0) {
      upstreamHeaders.set("Content-Length", String(contentLength));
    }

    const upstream = await fetch(BACKEND_URL, {
      method: "POST",
      headers: upstreamHeaders,
      body: request.body,
      duplex: "half",
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
    
    return await forwardUpstreamResponse(upstream);
  } catch (error) {
    console.error('Analysis API error:', error);
    
    if (error?.name === "AbortError" || error?.name === "TimeoutError") {
      return Response.json(
        {
          error: "Upload forwarding timed out before the backend responded.",
          code: "TIMEOUT",
          timeout: REQUEST_TIMEOUT_MS,
          detail: "The Python backend may still be processing if it already accepted the upload.",
        },
        { status: 504 }
      );
    }
    
    return Response.json(
      {
        error: error instanceof Error ? error.message : "Unable to reach the TRIBE backend.",
        detail: "Check that the Python backend is running and reachable.",
        code: "INTERNAL_ERROR",
        timestamp: new Date().toISOString(),
      },
      { status: 503 }
    );
  }
}

async function forwardUpstreamResponse(upstream) {
  const text = await upstream.text();
  const contentType =
    upstream.headers.get("content-type") ?? "application/json; charset=utf-8";

  if (upstream.ok || upstream.status === 202) {
    return new Response(text, {
      status: upstream.status,
      headers: {
        "Content-Type": contentType,
      },
    });
  }

  return Response.json(
    extractErrorPayload(text),
    { status: upstream.status }
  );
}

function extractErrorPayload(text) {
  try {
    const payload = JSON.parse(text);
    if (payload && typeof payload === "object") {
      return {
        error: String(payload.error ?? "Analysis backend returned an error."),
        detail:
          typeof payload.detail === "string"
            ? payload.detail
            : typeof payload.details === "string"
              ? payload.details
              : undefined,
        code: typeof payload.code === "string" ? payload.code : undefined,
      };
    }
  } catch {
    // Fall back to a generic message below.
  }

  return { error: "Analysis backend returned an error." };
}

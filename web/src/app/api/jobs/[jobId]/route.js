export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND_BASE_URL = (process.env.TRIBE_API_URL ?? "http://127.0.0.1:8002/analyze").replace(
  /\/analyze$/,
  ""
);
const REQUEST_TIMEOUT_MS = Number(process.env.TRIBE_REQUEST_TIMEOUT_MS ?? 30_000);

export async function GET(_request, { params }) {
  let jobId = "";
  try {
    ({ jobId } = await params);

    if (!jobId) {
      return Response.json(
        { 
          error: "Job ID is required.", 
          code: "MISSING_JOB_ID",
          details: "Please provide a valid job ID."
        }, 
        { status: 400 }
      );
    }

    if (!/^[a-zA-Z0-9_-]+$/.test(jobId)) {
      return Response.json(
        { 
          error: "Invalid job ID format.", 
          code: "INVALID_JOB_ID_FORMAT",
          details: "Job ID can only contain letters, numbers, underscores, and hyphens.",
          jobId: jobId
        }, 
        { status: 400 }
      );
    }

    if (jobId.length > 100) {
      return Response.json(
        { 
          error: "Job ID too long.", 
          code: "JOB_ID_TOO_LONG",
          details: "Job ID must be less than 100 characters.",
          maxLength: 100
        }, 
        { status: 400 }
      );
    }

    const upstream = await fetch(`${BACKEND_BASE_URL}/jobs/${jobId}`, {
      method: "GET",
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
    
    return await forwardUpstreamResponse(upstream);
  } catch (error) {
    console.error(`Jobs API error for job ID:`, jobId, error);
    
    if (error?.name === "AbortError" || error?.name === "TimeoutError") {
      return Response.json(
        {
          error: "Request timeout while checking job status.",
          code: "TIMEOUT",
          timeout: REQUEST_TIMEOUT_MS,
        },
        { status: 408 }
      );
    }
    
    return Response.json(
      {
        error: error instanceof Error ? error.message : "Unable to reach TRIBE backend.",
        detail: "Check that the Python backend is still running.",
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

  if (upstream.ok) {
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

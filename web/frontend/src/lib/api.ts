const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export interface PSFResponse {
  psf: number[][];
  preview_png: string;
  sharpness: number;
  psf_crop: number;
}

export interface CaseOut {
  id: string;
  mode: "simulate" | "diagnose";
  source_image_url: string;
  blurred_image_url: string | null;
  predistorted_image_url: string | null;
  S: number | null;
  C: number | null;
  axis: number | null;
  M: number;
  J0: number;
  J45: number;
  confidence: number | null;
  share_slug: string;
  created_at: string;
}

export interface SimulateResponse {
  case: CaseOut;
  shader_kernel: number[][];
}

export interface DiagnoseResponse {
  case: CaseOut;
  shader_kernel: number[][];
  confidence_detail: Record<string, number>;
  disclaimer: string;
}

export interface GalleryPage {
  items: CaseOut[];
  total: number;
  limit: number;
  offset: number;
}

export interface StatsResponse {
  total_cases: number;
  simulate_count: number;
  diagnose_count: number;
  m_histogram: { bucket: string; count: number }[];
  avg_diagnose_confidence: number | null;
}

function resolveImageUrl(url: string): string {
  return url.startsWith("/static/") ? `${API_BASE}${url}` : url;
}

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export async function getPsf(
  S: number,
  C: number,
  axis: number,
  pupilRMm = 2.0,
): Promise<PSFResponse> {
  const params = new URLSearchParams({
    S: String(S), C: String(C), axis: String(axis), pupil_r_mm: String(pupilRMm),
  });
  return asJson(await fetch(`${API_BASE}/api/psf?${params}`));
}

export async function postSimulate(
  image: File, S: number, C: number, axis: number, isPublic = true,
): Promise<SimulateResponse> {
  const form = new FormData();
  form.append("image", image);
  form.append("S", String(S));
  form.append("C", String(C));
  form.append("axis", String(axis));
  form.append("is_public", String(isPublic));
  const data = await asJson<SimulateResponse>(
    await fetch(`${API_BASE}/api/cases/simulate`, { method: "POST", body: form }),
  );
  return withResolvedUrls(data);
}

export async function postDiagnose(image: File, isPublic = true): Promise<DiagnoseResponse> {
  const form = new FormData();
  form.append("image", image);
  form.append("is_public", String(isPublic));
  const data = await asJson<DiagnoseResponse>(
    await fetch(`${API_BASE}/api/cases/diagnose`, { method: "POST", body: form }),
  );
  return withResolvedUrls(data);
}

function withResolvedUrls<T extends { case: CaseOut }>(data: T): T {
  const c = data.case;
  return {
    ...data,
    case: {
      ...c,
      source_image_url: resolveImageUrl(c.source_image_url),
      blurred_image_url: c.blurred_image_url ? resolveImageUrl(c.blurred_image_url) : null,
      predistorted_image_url: c.predistorted_image_url
        ? resolveImageUrl(c.predistorted_image_url) : null,
    },
  };
}

export async function getGallery(
  opts: { mode?: "simulate" | "diagnose"; limit?: number; offset?: number } = {},
): Promise<GalleryPage> {
  const params = new URLSearchParams();
  if (opts.mode) params.set("mode", opts.mode);
  params.set("limit", String(opts.limit ?? 24));
  params.set("offset", String(opts.offset ?? 0));
  const page = await asJson<GalleryPage>(await fetch(`${API_BASE}/api/gallery?${params}`));
  return { ...page, items: page.items.map((c) => withResolvedUrls({ case: c }).case) };
}

export async function getStats(): Promise<StatsResponse> {
  return asJson(await fetch(`${API_BASE}/api/stats`));
}

export async function getCaseBySlug(slug: string): Promise<CaseOut> {
  const c = await asJson<CaseOut>(await fetch(`${API_BASE}/api/share/${slug}`));
  return withResolvedUrls({ case: c }).case;
}

export function shareCardUrl(slug: string): string {
  return `${API_BASE}/api/share/${slug}/card.png`;
}

export { API_BASE };

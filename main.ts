// FreeModel-2 — a live free-model aggregator for make.fun's free-model-aggregator-02.
//
// One server, two client wires (OpenAI Chat Completions, Anthropic Messages),
// one upstream wire (the grader's gateway, Chat Completions only) and fourteen
// free OpenRouter models behind it. For each of the trial's six prompts it
// picks the highest-rated free model that meets the prompt's requirements,
// dials it through the gateway, and fails over to the next-best free model
// when the gateway refuses (429/5xx/401-per-model). Every answer comes back in
// the wire the client used, JSON or server-sent events, with the provider's
// text whole, its usage counts as reported, and its stop reason mapped — and
// with the `model` field naming the model really dialled, on every answer and
// every streamed chunk, because the house scores the model its own gateway
// record shows answering.
//
// Routing, in one paragraph: quality is 60% of the score, so candidates are
// ranked by the brief's own rating_milli, best first, ties broken by model id
// — a pure function of the brief, so a live pass and its replay route the
// same way. A refusal is never taken as proof of a spent budget: a 429 can be
// a one-off on the provider's schedule, so the model stays eligible for later
// prompts and is only skipped for the prompt it refused. When every candidate
// has refused a prompt once, the three best get one more try each before the
// prompt is given up. Every refusal is reported on the answer in the
// x-makefun-upstream-failures header, in order, so the failover column can see
// a refusal the server retried and then served.
//
// Determinism: no wall clock and no unseeded randomness anywhere a response
// can read — fixed timestamps, fixed ids, routing a pure function of the brief,
// the manifest and the gateway's answers — so a replay leaves the same state
// export byte for byte. The catalog is built once from the brief, so `--list`
// prints the same document the register body carries, byte for byte, with or
// without a live env.
//
// Node 22, standard library only — nothing is installed before this runs.

import { createServer } from 'node:http';
import type { IncomingMessage, Server, ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

// ---------------------------------------------------------------------------
// Small value helpers.
// ---------------------------------------------------------------------------

type AnyObj = Record<string, unknown>;

const isObj = (v: unknown): v is AnyObj => v !== null && typeof v === 'object' && !Array.isArray(v);
const asText = (v: unknown): string => (typeof v === 'string' ? v : '');
const asInt = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const asFlag = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

function log(message: string): void {
  process.stderr.write(`freemodel2: ${message}\n`);
}

// ---------------------------------------------------------------------------
// Fixed points: protocol, timing, ids. Nothing here reads the clock.
// ---------------------------------------------------------------------------

/** The register body's protocol version. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_MS = 100;
/** One gateway call's ceiling; the env allows 30 s a prompt, this leaves room to fail over. */
const GATEWAY_TIMEOUT_MS = 12000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The gateway refuses a call with no output cap, or one over its cap. */
const DEFAULT_MAX_TOKENS = 512;
const MAX_OUTPUT_TOKENS = 8192;
/** Fixed `created` for every OpenAI-shaped document this server emits. */
const FIXED_CREATED = 1767225600;
/** Fixed `created_at` for every Anthropic-shaped model listing. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
/** Fixed completion ids, so no run mints a fresh one. */
const CHAT_ID = 'chatcmpl-fma2';
const MESSAGE_ID = 'msg_fma2';
/** The header a served answer reports its upstream refusals on. */
const FAILURES_HEADER = 'x-makefun-upstream-failures';
const MAX_TRAIL_ROWS = 8;
const MAX_TRAIL_BYTES = 8192;
/** Second pass: how many of the best candidates get one more try after all refused. */
const SECOND_PASS_TOP = 3;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const BRIEF_PATH = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_BASE = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

// ---------------------------------------------------------------------------
// The brief: two JSON documents found by their `schema` member.
// ---------------------------------------------------------------------------

interface FreeModel {
  /** The id the gateway routes: openrouter/<vendor>/<model>:free. */
  id: string;
  provider: string;
  label: string;
  /** The house's quality rating in milli-points: the quality column's currency. */
  rating: number;
  context: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

interface PromptNeed {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

interface PromptSpec {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  needs: PromptNeed;
}

interface Brief {
  models: FreeModel[];
  prompts: PromptSpec[];
  problem: string | null;
}

/** The index of the `}` closing the object that opens at `open`, or -1; strings are skipped. */
function braceClose(text: string, open: number): number {
  let depth = 0;
  let inString = false;
  for (let i = open; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in order of appearance. */
function schemaDocs(text: string): AnyObj[] {
  const out: AnyObj[] = [];
  for (let open = text.indexOf('{'); open !== -1; open = text.indexOf('{', open + 1)) {
    const close = braceClose(text, open);
    if (close === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(open, close + 1));
      if (isObj(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/**
 * The first document under this contest's schema prefix that carries one of
 * `keys` as an array. A brief that spells its lists in the sibling contest's
 * names still reads: a document is found by the arrays it carries, never by a
 * name alone.
 */
function arrayDoc(docs: readonly AnyObj[], keys: readonly string[]): { doc: AnyObj; key: string } | null {
  for (const doc of docs) {
    if (!asText(doc.schema).includes('free-model-aggregator-02/')) continue;
    for (const key of keys) if (Array.isArray(doc[key])) return { doc, key };
  }
  return null;
}

/**
 * One model row. The id is the row's `id`, or its `model` under the provider
 * that names it: a row that gives the bare OpenRouter id (`vendor/model:free`)
 * and names `openrouter` is routed as the gateway spells it
 * (`openrouter/vendor/model:free`). A row is free unless it says
 * `free: false`; a paid row is never routed.
 */
function modelRowOf(v: unknown): FreeModel | null {
  if (!isObj(v)) return null;
  const raw = asText(v.id) !== '' ? asText(v.id) : asText(v.model);
  if (raw === '') return null;
  const named = asText(v.provider) !== '' ? asText(v.provider) : asText(v.wire);
  const id = named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    label: asText(v.display_name) !== '' ? asText(v.display_name) : raw,
    rating: asInt(v.rating_milli) ?? 0,
    context: asInt(v.context_length),
    tools: asFlag(v.tools),
    vision: asFlag(v.vision),
    json: asFlag(v.json),
    free: v.free !== false,
  };
}

function promptSpecOf(v: unknown): PromptSpec | null {
  if (!isObj(v)) return null;
  const id = asText(v.id);
  if (id === '') return null;
  const req = isObj(v.requires) ? v.requires : {};
  const needs: PromptNeed = {};
  if (req.tools === true) needs.tools = true;
  if (req.vision === true) needs.vision = true;
  if (req.json === true) needs.json = true;
  if (typeof req.min_context === 'number' && Number.isFinite(req.min_context)) needs.minContext = req.min_context;
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    needs,
  };
}

/** The brief at `path`; an empty brief with `problem` set when it cannot be read at all. */
function readBrief(path: string): Brief {
  const empty: Brief = { models: [], prompts: [], problem: null };
  if (path === '') return { ...empty, problem: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, problem: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocs(text);
  const modelsDoc = arrayDoc(docs, ['models', 'ratings']);
  if (modelsDoc === null) {
    return { ...empty, problem: `the brief at ${path} holds no free-model-aggregator-02/models/v1 document` };
  }
  const models = (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((v) => {
    const m = modelRowOf(v);
    return m === null ? [] : [m];
  });
  const promptsDoc = arrayDoc(docs, ['prompts', 'manifest']);
  const prompts =
    promptsDoc === null || !Array.isArray(promptsDoc.doc[promptsDoc.key])
      ? []
      : (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((v) => {
          const p = promptSpecOf(v);
          return p === null ? [] : [p];
        });
  return { models, prompts, problem: null };
}

// ---------------------------------------------------------------------------
// The catalog: one provider, exactly the brief's free models.
// ---------------------------------------------------------------------------

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface Catalog {
  providers: { id: string; base_url: string; key_env: string; wire: 'openai'; models: CatalogModel[] }[];
}

/**
 * The one catalog: the brief's free models, deduplicated, sorted by id so the
 * document is a pure function of the brief's set, not of the order its rows
 * arrived in. The base_url is OpenRouter's public endpoint, never the
 * gateway's trial-local address: the gateway's origin differs between a live
 * pass and its replay, and the catalog is part of the export a replay must
 * reproduce byte for byte.
 */
function buildCatalog(brief: Brief): Catalog {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of [...brief.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.context });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: 'https://openrouter.ai/api/v1',
        key_env: 'OPENROUTER_API_KEY',
        wire: 'openai',
        models,
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// Routing: the best free model that meets a request.
// ---------------------------------------------------------------------------

/** Whether a model's capabilities meet one prompt's requirements. Unknown never meets one. */
function meetsNeed(need: PromptNeed, model: FreeModel): boolean {
  if (need.tools === true && model.tools !== true) return false;
  if (need.vision === true && model.vision !== true) return false;
  if (need.json === true && model.json !== true) return false;
  if (need.minContext !== undefined && (model.context === null || model.context < need.minContext)) return false;
  return true;
}

/**
 * Every free model that can serve `need`, best rating first. Ties break by
 * model id, so the order is a pure function of the brief — the same on a live
 * pass and its replay.
 */
function rankedCandidates(brief: Brief, need: PromptNeed): FreeModel[] {
  return brief.models
    .filter((m) => m.free && meetsNeed(need, m))
    .sort((a, b) => b.rating - a.rating || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

/** The free model ids this server serves, in catalog order — exactly the registered free ids. */
function servedIds(brief: Brief): { id: string; label: string }[] {
  const seen = new Set<string>();
  const out: { id: string; label: string }[] = [];
  for (const model of [...brief.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    out.push({ id: model.id, label: model.label });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Client requests, read in the wire they arrived in.
// ---------------------------------------------------------------------------

/** The text of a string or of a block array's text blocks; '' when there is none. */
function blockText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .filter((b): b is AnyObj => isObj(b) && b.type === 'text')
    .map((b) => asText(b.text))
    .join('\n');
}

/** Whether any content part of a message is an image, in either wire's spelling. */
function contentHasImage(content: unknown): boolean {
  if (!Array.isArray(content)) return false;
  return content.some(
    (b) => isObj(b) && (b.type === 'image' || b.type === 'image_url'),
  );
}

/** The requirements a request body carries on its own face, when the brief has no row for it. */
function needsFromBody(wire: 'openai' | 'anthropic', body: AnyObj): PromptNeed {
  const messages = Array.isArray(body.messages) ? body.messages.filter((m): m is AnyObj => isObj(m)) : [];
  let vision = false;
  for (const message of messages) if (contentHasImage(message.content)) vision = true;
  const format = isObj(body.response_format) ? body.response_format : null;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format !== null && (format.type === 'json_object' || format.type === 'json_schema'),
  };
}

// ---------------------------------------------------------------------------
// Translation: the client's wire -> the gateway's Chat Completions body.
// ---------------------------------------------------------------------------

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function anthropicPart(block: AnyObj): AnyObj | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isObj(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blockText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** An Anthropic Messages request as a Chat Completions request naming `model`. */
function anthropicToChat(body: AnyObj, model: string): AnyObj {
  const messages: AnyObj[] = [];
  const system = blockText(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages.filter((m): m is AnyObj => isObj(m)) : [];
  for (const raw of source) {
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: raw.role === 'assistant' ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (raw.role === 'assistant') {
      const toolCalls: AnyObj[] = [];
      for (const block of content) {
        if (!isObj(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: asText(block.id),
          type: 'function',
          function: { name: asText(block.name), arguments: JSON.stringify(isObj(block.input) ? block.input : {}) },
        });
      }
      const text = blockText(content);
      const assistant: AnyObj = { role: 'assistant', content: text !== '' ? text : null };
      if (toolCalls.length > 0) assistant.tool_calls = toolCalls;
      messages.push(assistant);
      continue;
    }
    const parts: AnyObj[] = [];
    for (const block of content) {
      if (!isObj(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: asText(block.tool_use_id), content: toolResultText(block.content) });
        continue;
      }
      const part = anthropicPart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: AnyObj = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isObj(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: asText(raw.description),
            parameters: isObj(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isObj(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** An OpenAI Chat Completions request as the gateway wants it: the model pinned, the cap enforced. */
function openAIToChat(body: AnyObj, model: string): AnyObj {
  const out: AnyObj = { ...body, model };
  const cap = asInt(body.max_tokens) ?? asInt(body.max_completion_tokens);
  out.max_tokens = cap === null ? DEFAULT_MAX_TOKENS : Math.min(cap, MAX_OUTPUT_TOKENS);
  if (body.stream === true) {
    const options = isObj(body.stream_options) ? body.stream_options : {};
    out.stream_options = { ...options, include_usage: true };
  }
  return out;
}

/** The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route. */
function gatewayChatUrl(): string {
  if (GATEWAY_BASE === '') return '';
  return GATEWAY_BASE.endsWith('/v1') ? `${GATEWAY_BASE}/chat/completions` : `${GATEWAY_BASE}/v1/chat/completions`;
}

// ---------------------------------------------------------------------------
// Upstream calls: one dial, one refusal row when it fails.
// ---------------------------------------------------------------------------

interface Refusal {
  status: number;
  code: string;
  model: string | null;
}

interface DialResult {
  ok: boolean;
  /** HTTP status; 502 when the call never reached the gateway at all. */
  status: number;
  code: string;
  text: string;
}

/** The refusal code of a gateway error body, or a status-derived one when the body names none. */
function refusalCode(text: string, status: number): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isObj(value) && isObj(value.error)) {
      if (typeof value.error.code === 'string' && value.error.code !== '') return value.error.code.slice(0, 64);
      if (typeof value.error.code === 'number') return String(value.error.code);
      if (typeof value.error.type === 'string' && value.error.type !== '') return value.error.type.slice(0, 64);
    }
  } catch {
    // Not JSON: fall through to the status-derived code.
  }
  return status === 429
    ? 'rate_limited'
    : status === 401
      ? 'unauthorized'
      : status === 400
        ? 'bad_request'
        : status >= 500
          ? 'upstream_error'
          : `http_${status}`;
}

/** One call to one free model on the gateway's Chat Completions route. */
async function dialGateway(modelId: string, chatBody: AnyObj, stream: boolean): Promise<DialResult> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (stream) headers.accept = 'text/event-stream';
  let res: Response;
  try {
    res = await fetch(gatewayChatUrl(), {
      method: 'POST',
      headers,
      body: JSON.stringify(chatBody),
      signal: AbortSignal.timeout(GATEWAY_TIMEOUT_MS),
    });
  } catch (err) {
    const timedOut = (err as Error).name === 'TimeoutError';
    return { ok: false, status: 502, code: timedOut ? 'timeout' : 'unreachable', text: '' };
  }
  const text = await res.text();
  if (!res.ok) return { ok: false, status: res.status, code: refusalCode(text, res.status), text };
  return { ok: true, status: res.status, code: '', text };
}

// ---------------------------------------------------------------------------
// Provider answers, parsed in the wire they arrived in, re-emitted in the
// wire the client used. `stop` is normalized to OpenAI's four stop reasons;
// each emitter maps it to its own wire's spelling. Streams are buffered whole
// and re-emitted, so a cross-wire answer keeps the provider's text, usage and
// stop reason exactly — and message_start can carry the true input count.
// ---------------------------------------------------------------------------

type Stop = 'stop' | 'length' | 'tool_calls' | 'content_filter';

interface ToolCallOut {
  id: string;
  name: string;
  args: string;
}

interface Completion {
  text: string;
  toolCalls: ToolCallOut[];
  inputTokens: number;
  outputTokens: number;
  stop: Stop;
}

function openAIStopOf(value: unknown): Stop {
  if (value === 'length') return 'length';
  if (value === 'tool_calls') return 'tool_calls';
  if (value === 'content_filter') return 'content_filter';
  return 'stop';
}

const anthropicStopOf = (stop: Stop): string =>
  stop === 'length' ? 'max_tokens' : stop === 'tool_calls' ? 'tool_use' : stop === 'content_filter' ? 'refusal' : 'end_turn';

/** A Chat Completions JSON answer as one Completion, or null when it does not parse. */
function parseChatJson(text: string): Completion | null {
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isObj(body) || !Array.isArray(body.choices) || body.choices.length === 0) return null;
  const choice = body.choices[0];
  if (!isObj(choice) || !isObj(choice.message)) return null;
  const message = choice.message;
  const usage = isObj(body.usage) ? body.usage : {};
  const toolCalls: ToolCallOut[] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isObj(call) || !isObj(call.function)) continue;
      toolCalls.push({
        id: asText(call.id),
        name: asText(call.function.name),
        args: asText(call.function.arguments),
      });
    }
  }
  return {
    text: typeof message.content === 'string' ? message.content : blockText(message.content),
    toolCalls,
    inputTokens: asInt(usage.prompt_tokens) ?? 0,
    outputTokens: asInt(usage.completion_tokens) ?? 0,
    stop: openAIStopOf(choice.finish_reason),
  };
}

/** One SSE frame's `data:` lines joined; null when the frame carries no data. */
function frameData(frame: string): string | null {
  const data: string[] = [];
  for (const line of frame.split(/\r?\n/)) {
    if (line.startsWith('data:')) data.push(line.slice(5).trimStart());
    else if (line.startsWith(':')) continue;
    else return null;
  }
  return data.length > 0 ? data.join('\n') : null;
}

/** A Chat Completions SSE body as one Completion, or null when it does not parse. */
function parseChatStream(text: string): Completion | null {
  let content = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: Stop = 'stop';
  let sawAny = false;
  const toolParts = new Map<number, { id: string; name: string; args: string }>();
  for (const frame of text.split(/\r?\n\r?\n/)) {
    const data = frameData(frame.trim());
    if (data === null || data === '[DONE]') continue;
    let event: unknown;
    try {
      event = JSON.parse(data);
    } catch {
      return null;
    }
    if (!isObj(event)) return null;
    sawAny = true;
    if (isObj(event.usage)) {
      inputTokens = asInt(event.usage.prompt_tokens) ?? inputTokens;
      outputTokens = asInt(event.usage.completion_tokens) ?? outputTokens;
    }
    if (!Array.isArray(event.choices) || event.choices.length === 0) continue;
    const choice = event.choices[0];
    if (!isObj(choice)) return null;
    if (isObj(choice.delta)) {
      const delta = choice.delta;
      if (typeof delta.content === 'string') content += delta.content;
      if (Array.isArray(delta.tool_calls)) {
        for (const call of delta.tool_calls) {
          if (!isObj(call)) continue;
          const index = asInt(call.index) ?? toolParts.size;
          const part = toolParts.get(index) ?? { id: '', name: '', args: '' };
          if (typeof call.id === 'string') part.id = call.id;
          if (isObj(call.function)) {
            if (typeof call.function.name === 'string') part.name = call.function.name;
            if (typeof call.function.arguments === 'string') part.args += call.function.arguments;
          }
          toolParts.set(index, part);
        }
      }
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
      stop = openAIStopOf(choice.finish_reason);
    }
  }
  if (!sawAny) return null;
  return {
    text: content,
    toolCalls: [...toolParts.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([, part]) => ({ id: part.id, name: part.name, args: part.args })),
    inputTokens,
    outputTokens,
    stop,
  };
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argsObject(value: string): AnyObj {
  if (value.trim() === '') return {};
  try {
    const parsed: unknown = JSON.parse(value);
    if (isObj(parsed)) return parsed;
  } catch {
    // Not JSON: an empty input.
  }
  return {};
}

/** One Completion as a Chat Completions JSON answer naming `model`. */
function toChatJson(answer: Completion, model: string): AnyObj {
  const message: AnyObj = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((call) => ({
      id: call.id,
      type: 'function',
      function: { name: call.name, arguments: call.args },
    }));
  }
  return {
    id: CHAT_ID,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: answer.stop }],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** One Completion as an Anthropic Messages answer naming `model`. */
function toMessagesJson(answer: Completion, model: string): AnyObj {
  const content: AnyObj[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({
      type: 'tool_use',
      id: call.id !== '' ? call.id : 'toolu_unknown',
      name: call.name,
      input: argsObject(call.args),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: MESSAGE_ID,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStopOf(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** Split text into pieces of at most `size` characters; at least one piece, even for empty text. */
function splitPieces(text: string, size: number): string[] {
  const out: string[] = [];
  for (let i = 0; i < text.length; i += size) out.push(text.slice(i, i + size));
  if (out.length === 0) out.push('');
  return out;
}

/** One Completion as a Chat Completions SSE stream naming `model`, in the order the env's checks read: content chunks, one finish chunk carrying usage, then [DONE]. */
function toChatStream(answer: Completion, model: string): string {
  const frames: string[] = [];
  const head: AnyObj = {
    id: CHAT_ID,
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }],
  };
  frames.push(`data: ${JSON.stringify(head)}\n\n`);
  for (const piece of splitPieces(answer.text, 64)) {
    const chunk: AnyObj = {
      id: CHAT_ID,
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, delta: { content: piece }, finish_reason: null }],
    };
    frames.push(`data: ${JSON.stringify(chunk)}\n\n`);
  }
  for (let i = 0; i < answer.toolCalls.length; i++) {
    const call = answer.toolCalls[i];
    const chunk: AnyObj = {
      id: CHAT_ID,
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [
              {
                index: i,
                id: call.id,
                type: 'function',
                function: { name: call.name, arguments: call.args },
              },
            ],
          },
          finish_reason: null,
        },
      ],
    };
    frames.push(`data: ${JSON.stringify(chunk)}\n\n`);
  }
  const tail: AnyObj = {
    id: CHAT_ID,
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta: {}, finish_reason: answer.stop }],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
  frames.push(`data: ${JSON.stringify(tail)}\n\n`);
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** One SSE event: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One Completion as an Anthropic SSE stream naming `model`: message_start, one text block (and one tool_use block a call), message_delta with the stop reason and output count, then message_stop. */
function toMessagesStream(answer: Completion, model: string): string {
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: MESSAGE_ID,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  frames.push(
    sseEvent('content_block_start', {
      type: 'content_block_start',
      index,
      content_block: { type: 'text', text: '' },
    }),
  );
  for (const piece of splitPieces(answer.text, 64)) {
    frames.push(
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index,
        delta: { type: 'text_delta', text: piece },
      }),
    );
  }
  frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
  index += 1;
  for (const call of answer.toolCalls) {
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: {
          type: 'tool_use',
          id: call.id !== '' ? call.id : `toolu_${index}`,
          name: call.name,
          input: {},
        },
      }),
    );
    if (call.args !== '') {
      frames.push(
        sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: { type: 'input_json_delta', partial_json: call.args },
        }),
      );
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopOf(answer.stop), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
  );
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}

// ---------------------------------------------------------------------------
// Errors, in the wire the client used.
// ---------------------------------------------------------------------------

function openAIError(message: string, type: string): AnyObj {
  return { error: { message, type, code: null } };
}

function anthropicError(type: string, message: string): AnyObj {
  return { type: 'error', error: { type, message } };
}

function sendJson(res: ServerResponse, status: number, value: AnyObj, failures: Refusal[] = []): void {
  const text = JSON.stringify(value);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'content-length': String(Buffer.byteLength(text)),
  };
  const trail = trailHeader(failures);
  if (trail !== null) headers[FAILURES_HEADER] = trail;
  res.writeHead(status, headers);
  res.end(text);
}

function sendStream(res: ServerResponse, text: string, failures: Refusal[] = []): void {
  const headers: Record<string, string> = {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'content-length': String(Buffer.byteLength(text)),
  };
  const trail = trailHeader(failures);
  if (trail !== null) headers[FAILURES_HEADER] = trail;
  res.writeHead(200, headers);
  res.end(text);
}

/**
 * The refusals trail for one answer: at most 8 rows, in the order the calls
 * refused, and under 8 KiB — oldest rows drop first when it would not fit.
 * An empty trail is no header at all.
 */
function trailHeader(failures: Refusal[]): string | null {
  if (failures.length === 0) return null;
  const rows = failures.slice(-MAX_TRAIL_ROWS).map((f) => ({ status: f.status, code: f.code, model: f.model }));
  let text = JSON.stringify(rows);
  while (rows.length > 0 && Buffer.byteLength(text) > MAX_TRAIL_BYTES) {
    rows.shift();
    text = JSON.stringify(rows);
  }
  return rows.length === 0 ? null : text;
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        req.destroy();
        reject(new Error('request body too large'));
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

// ---------------------------------------------------------------------------
// One prompt: route, dial, fail over, answer in the client's wire.
// ---------------------------------------------------------------------------

interface ServerState {
  brief: Brief;
  catalog: Catalog;
  /** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
  manifestIndex: number;
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(state: ServerState, wire: 'openai' | 'anthropic', body: AnyObj): PromptNeed {
  const entry = state.brief.prompts[state.manifestIndex];
  state.manifestIndex += 1;
  if (entry !== undefined) return entry.needs;
  return needsFromBody(wire, body);
}

/**
 * Walk the candidates best-first and stop at the first model that answers
 * validly. A refusal is recorded and the walk moves on; a 401 ends the walk
 * at once, because the key — not the model — is what the gateway refused.
 * When every candidate has refused once, the three best get one more try
 * each before the prompt is given up.
 */
async function serveWithFailover(
  state: ServerState,
  wire: 'openai' | 'anthropic',
  body: AnyObj,
  need: PromptNeed,
): Promise<{ answer: Completion; model: string; failures: Refusal[] } | { answer: null; failures: Refusal[] }> {
  const stream = body.stream === true;
  const candidates = rankedCandidates(state.brief, need);
  const failures: Refusal[] = [];
  const attempts: FreeModel[] = [...candidates, ...candidates.slice(0, SECOND_PASS_TOP)];
  for (const candidate of attempts) {
    const chatBody = wire === 'anthropic' ? anthropicToChat(body, candidate.id) : openAIToChat(body, candidate.id);
    const result = await dialGateway(candidate.id, chatBody, stream);
    if (!result.ok) {
      log(`  ${candidate.id} refused (${result.status} ${result.code})`);
      failures.push({ status: result.status, code: result.code, model: candidate.id });
      if (result.status === 401) break;
      continue;
    }
    const parsed = stream ? parseChatStream(result.text) : parseChatJson(result.text);
    if (parsed === null) {
      log(`  ${candidate.id} answered unreadably`);
      failures.push({ status: result.status, code: 'unparseable_answer', model: candidate.id });
      continue;
    }
    log(`  served by ${candidate.id} (rating ${candidate.rating})`);
    return { answer: parsed, model: candidate.id, failures };
  }
  return { answer: null, failures };
}

/** One chat/messages request: route, fail over, answer in the client's wire. */
async function handleChat(
  state: ServerState,
  wire: 'openai' | 'anthropic',
  req: IncomingMessage,
  res: ServerResponse,
): Promise<void> {
  let body: AnyObj;
  try {
    const text = await readBody(req);
    const parsed: unknown = JSON.parse(text);
    if (!isObj(parsed)) throw new Error('body is not an object');
    body = parsed;
  } catch (err) {
    const message = `invalid request body: ${(err as Error).message}`;
    if (wire === 'anthropic') sendJson(res, 400, anthropicError('invalid_request_error', message));
    else sendJson(res, 400, openAIError(message, 'invalid_request_error'));
    return;
  }
  const index = state.manifestIndex;
  const stream = body.stream === true;
  const need = requirementsFor(state, wire, body);
  log(`request ${index + 1} (${wire}${stream ? ', stream' : ''}): ${rankedCandidates(state.brief, need).length} candidates`);
  const served = await serveWithFailover(state, wire, body, need);
  if (served.answer === null) {
    log(`  no free model could serve request ${index + 1}`);
    const message = 'no free model could serve this request';
    if (wire === 'anthropic') sendJson(res, 503, anthropicError('overloaded_error', message), served.failures);
    else sendJson(res, 503, openAIError(message, 'upstream_error'), served.failures);
    return;
  }
  const { answer, model, failures } = served;
  if (stream) {
    const text = wire === 'anthropic' ? toMessagesStream(answer, model) : toChatStream(answer, model);
    sendStream(res, text, failures);
  } else if (wire === 'anthropic') {
    sendJson(res, 200, toMessagesJson(answer, model), failures);
  } else {
    sendJson(res, 200, toChatJson(answer, model), failures);
  }
}

/** GET /v1/models: exactly the free models the catalog lists, in its wire's shape. */
function handleModels(state: ServerState, req: IncomingMessage, res: ServerResponse): void {
  const models = servedIds(state.brief);
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.label, created_at: FIXED_CREATED_AT })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: 'openrouter' })),
    });
  }
}

function createAggregator(state: ServerState): Server {
  return createServer((req, res) => {
    void (async () => {
      const url = req.url ?? '/';
      const path = url.split('?')[0];
      try {
        if (req.method === 'GET' && path === '/v1/models') {
          handleModels(state, req, res);
        } else if (req.method === 'GET' && (path === '/' || path === '/health')) {
          sendJson(res, 200, { ok: true });
        } else if (req.method === 'POST' && path === '/v1/chat/completions') {
          await handleChat(state, 'openai', req, res);
        } else if (req.method === 'POST' && path === '/v1/messages') {
          await handleChat(state, 'anthropic', req, res);
        } else {
          sendJson(res, 404, openAIError(`unknown route ${req.method ?? ''} ${path}`, 'not_found'));
        }
      } catch (err) {
        log(`handler error: ${(err as Error).message}`);
        if (!res.headersSent) sendJson(res, 500, openAIError('internal error', 'internal_error'));
      }
    })();
  });
}

// ---------------------------------------------------------------------------
// Register, poll, run.
// ---------------------------------------------------------------------------

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, catalog: Catalog): Promise<{ requests: number }> {
  let lastError = 'no attempt made';
  for (let attempt = 1; attempt <= 8; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(GATEWAY_TIMEOUT_MS),
      });
      const text = await res.text();
      if (!res.ok) {
        lastError = `register answered HTTP ${res.status}: ${text.slice(0, 200)}`;
      } else {
        const parsed: unknown = JSON.parse(text);
        if (isObj(parsed) && parsed.ok === true) {
          return { requests: typeof parsed.requests === 'number' ? parsed.requests : 0 };
        }
        lastError = `register answered ${text.slice(0, 200)}`;
      }
    } catch (err) {
      lastError = (err as Error).message;
    }
    if (attempt < 8) await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(lastError);
}

/** Poll GET /work until it says done, then stop the server and exit 0. */
async function pollWork(server: Server): Promise<void> {
  for (;;) {
    try {
      const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(GATEWAY_TIMEOUT_MS) });
      if (res.ok) {
        const parsed: unknown = await res.json();
        if (isObj(parsed) && parsed.done === true) {
          server.close();
          process.exit(0);
        }
      }
    } catch {
      // The env answers again on the next poll.
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_MS));
  }
}

async function start(): Promise<void> {
  const args = process.argv.slice(2);
  const brief = readBrief(BRIEF_PATH);
  const catalog = buildCatalog(brief);
  if (args.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts. With no brief to
    // read the catalog is empty, and that empty document is what prints.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (brief.problem !== null) {
    log(brief.problem);
    process.exit(1);
  }
  if (brief.models.length === 0) {
    log('the brief holds no free models');
    process.exit(1);
  }
  if (ENV_URL === '') {
    log('no MAKEFUN_ENV_URL: serving without registering');
    const state: ServerState = { brief, catalog, manifestIndex: 0 };
    const server = createAggregator(state);
    await new Promise<void>((resolve, reject) => {
      server.once('error', reject);
      server.listen(0, '0.0.0.0', () => resolve());
    });
    const address = server.address();
    log(`listening on 0.0.0.0:${typeof address === 'object' && address !== null ? address.port : 0} with ${servedIds(brief).length} free models`);
    return;
  }
  if (gatewayChatUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const state: ServerState = { brief, catalog, manifestIndex: 0 };
  const server = createAggregator(state);
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
  const address = server.address();
  const port = typeof address === 'object' && address !== null ? address.port : 0;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedIds(brief).length} free models`);
  const registration = await register(port, catalog);
  log(`registered: the trial holds ${registration.requests} prompts`);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  await pollWork(server);
}

start().catch((err: unknown) => {
  log(`fatal: ${(err as Error).message}`);
  process.exit(1);
});

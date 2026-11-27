// solve.js — a faster drop-in for makefun-diff 1.0.0.
//
// Same answer as makefun-diff on every input, faster: lines are interned to
// integer ids once, then Myers's greedy search (the header's rule: deletion
// preferred on a tie, no edge pruning, snakes followed to the end) runs over
// Int32Arrays with no per-step allocation, and the patch is formatted exactly
// as makefun-diff formats it.

const KEEP = '=';
const DELETE = '-';
const INSERT = '+';
const FORMAT = 'makefun-diff/1';
const CONTEXT = 3;

/** A text's lines, each with its "\n" (the last one without, when the text does not end in one). */
function tokenize(text) {
  const parts = text.split('\n');
  const lines = [];
  const last = parts.length - 1;
  for (let i = 0; i < last; i++) lines.push(parts[i] + '\n');
  if (parts[last] !== '') lines.push(parts[last]);
  return lines;
}

/**
 * The runs of the edit script of A into B: [{ kind, count }], in order.
 * A and B are Int32Arrays of interned line ids. The search is makefun-diff's
 * exactly: for d = 0, 1, 2, ... and each diagonal k = x - y from -d to d in
 * steps of 2, the furthest point is one old line past the (d-1)-point on
 * k - 1 (a deletion) or one new line past the (d-1)-point on k + 1 (an
 * insertion), whichever lies further into the old text, the deletion on a
 * tie; a move past the end of either text is no candidate; from that point
 * the path follows every line equal in both as far as it goes; the first
 * path to reach both ends is the script.
 */
function editRuns(A, B) {
  const N = A.length;
  const M = B.length;
  const maxD = N + M;
  // The V rows live in two compact buffers, reused every d: row d occupies
  // indices [0, 2d], x at (d, k) is buf[k + d], or -1 for no path. Both are
  // filled with -1 once; an index a row never wrote still reads -1, because
  // a buffer's earlier uses only ever wrote shorter prefixes of it.
  const W = 2 * maxD + 1;
  let prevC = new Int32Array(W).fill(-1);
  let curC = new Int32Array(W).fill(-1);
  // snaps[d] is the d-th row's written span, as a copy, so backtracking can
  // read any (d, k) after the search: snaps[d][k + d] is x at (d, k).
  // Measured: per-row slice() is the fastest snapshot store here (a flat
  // buffer with set(), a plain loop, and compact-buffer variants all ran
  // slower in this program, however fast the copy looks in isolation).
  const snaps = [];
  // d = 0: the snake from the start on diagonal 0.
  let x = 0;
  while (x < N && x < M && A[x] === B[x]) x++;
  prevC[0] = x;
  snaps.push(prevC.slice(0, 1));
  if (x >= N && x >= M) return [];
  let D = -1;
  let kFin = 0;
  for (let d = 1; d <= maxD; d++) {
    let done = false;
    for (let k = -d; k <= d; k += 2) {
      const idx = k + d;
      const pDown = idx >= 2 ? prevC[idx - 2] : -1;
      const pUp = prevC[idx];
      const canDel = pDown >= 0 && pDown < N;
      const canIns = pUp >= 0 && pUp - (k + 1) < M;
      if (!canDel && !canIns) {
        curC[idx] = -1;
        continue;
      }
      const xDel = canDel ? pDown + 1 : -1;
      const xIns = canIns ? pUp : -1;
      let xx = xDel >= xIns ? xDel : xIns;
      let yy = xx - k;
      while (xx < N && yy < M && A[xx] === B[yy]) {
        xx++;
        yy++;
      }
      curC[idx] = xx;
      if (xx >= N && yy >= M) {
        D = d;
        kFin = k;
        done = true;
        break;
      }
    }
    snaps.push(curC.slice(0, (done ? kFin + d : 2 * d) + 1));
    if (done) break;
    const tmp = prevC;
    prevC = curC;
    curC = tmp;
  }
  if (D < 0) throw new Error('makefun-diff: the search ended without reaching both ends');
  // Backtrack from (D, kFin): recompute each step's choice from the
  // snapshots (the same values the forward pass saw), collecting the edit
  // and the snake after it, then the d = 0 snake; reverse and merge.
  const rev = [];
  let k = kFin;
  for (let d = D; d >= 1; d--) {
    const row = snaps[d];
    const prevRow = snaps[d - 1];
    const xEnd = row[k + d];
    const iDown = k + d - 2;
    const iUp = k + d;
    const pDown = iDown >= 0 && iDown < prevRow.length ? prevRow[iDown] : -1;
    const pUp = iUp >= 0 && iUp < prevRow.length ? prevRow[iUp] : -1;
    const canDel = pDown >= 0 && pDown < N;
    const canIns = pUp >= 0 && pUp - (k + 1) < M;
    const xDel = canDel ? pDown + 1 : -1;
    const xIns = canIns ? pUp : -1;
    const takeDel = xDel >= xIns;
    const xEdit = takeDel ? xDel : xIns;
    const snake = xEnd - xEdit;
    if (snake > 0) rev.push({ kind: KEEP, count: snake });
    rev.push({ kind: takeDel ? DELETE : INSERT, count: 1 });
    k = takeDel ? k - 1 : k + 1;
  }
  const x0 = snaps[0][0];
  if (x0 > 0) rev.push({ kind: KEEP, count: x0 });
  const runs = [];
  for (let i = rev.length - 1; i >= 0; i--) {
    const r = rev[i];
    const last = runs[runs.length - 1];
    if (last !== undefined && last.kind === r.kind) last.count += r.count;
    else runs.push({ kind: r.kind, count: r.count });
  }
  return runs;
}

/** A line as the patch writes it: its op, the line, a newline, and "%" on a line of its own when it had none. */
function written(op, line) {
  return line.endsWith('\n') ? op + line : op + line + '\n%\n';
}

/** makefun-diff 1.0.0's answer for one case's input. */
export function run(input) {
  if (input === null || typeof input !== 'object' || typeof input.old !== 'string' || typeof input.new !== 'string') {
    throw new TypeError('makefun-diff: the input is { old, new }, two strings');
  }
  const a = tokenize(input.old);
  const b = tokenize(input.new);
  const n = a.length;
  const m = b.length;
  // Intern every line to an integer id: equality of lines becomes equality
  // of ids, and the search never compares strings again.
  const ids = new Map();
  let nextId = 0;
  const A = new Int32Array(n);
  const B = new Int32Array(m);
  for (let i = 0; i < n; i++) {
    let id = ids.get(a[i]);
    if (id === undefined) {
      id = nextId++;
      ids.set(a[i], id);
    }
    A[i] = id;
  }
  for (let i = 0; i < m; i++) {
    let id = ids.get(b[i]);
    if (id === undefined) {
      id = nextId++;
      ids.set(b[i], id);
    }
    B[i] = id;
  }
  // The common prefix, then the common suffix of what is left.
  let prefix = 0;
  while (prefix < n && prefix < m && A[prefix] === B[prefix]) prefix++;
  let suffix = 0;
  while (suffix < n - prefix && suffix < m - prefix && A[n - 1 - suffix] === B[m - 1 - suffix]) suffix++;
  const runs = [];
  if (prefix > 0) runs.push({ kind: KEEP, count: prefix });
  runs.push(...editRuns(A.subarray(prefix, n - suffix), B.subarray(prefix, m - suffix)));
  if (suffix > 0) runs.push({ kind: KEEP, count: suffix });

  // The change runs: each a stretch of deletions and insertions between two kept lines, at old line o and new line p.
  const changes = [];
  let o = 0;
  let p = 0;
  let deleted = 0;
  let inserted = 0;
  for (const r of runs) {
    if (r.kind === KEEP) {
      o += r.count;
      p += r.count;
      continue;
    }
    let c = changes.at(-1);
    if (c === undefined || c.o + c.dels !== o || c.p + c.ins !== p) {
      c = { o, p, dels: 0, ins: 0 };
      changes.push(c);
    }
    if (r.kind === DELETE) {
      c.dels += r.count;
      o += r.count;
      deleted += r.count;
    } else {
      c.ins += r.count;
      p += r.count;
      inserted += r.count;
    }
  }

  // The hunks: change runs at most 2 * CONTEXT kept lines apart share one.
  const hunks = [];
  for (const c of changes) {
    const h = hunks.at(-1);
    const prev = h?.at(-1);
    if (prev !== undefined && c.o - (prev.o + prev.dels) <= 2 * CONTEXT) h.push(c);
    else hunks.push([c]);
  }
  const out = [`${FORMAT} old=${n} new=${m} hunks=${hunks.length} deleted=${deleted} inserted=${inserted}\n`];
  for (const h of hunks) {
    const first = h[0];
    const last = h.at(-1);
    const oldStart = Math.max(0, first.o - CONTEXT);
    const newStart = first.p - (first.o - oldStart);
    const after = Math.min(CONTEXT, n - (last.o + last.dels));
    const oldEnd = last.o + last.dels + after;
    const newEnd = last.p + last.ins + after;
    out.push(`@ ${oldStart + 1} ${oldEnd - oldStart} ${newStart + 1} ${newEnd - newStart}\n`);
    let at = oldStart;
    for (const c of h) {
      for (; at < c.o; at++) out.push(written(KEEP, a[at]));
      for (let i = 0; i < c.dels; i++) out.push(written(DELETE, a[c.o + i]));
      for (let i = 0; i < c.ins; i++) out.push(written(INSERT, b[c.p + i]));
      at = c.o + c.dels;
    }
    for (; at < oldEnd; at++) out.push(written(KEEP, a[at]));
  }
  return out.join('');
}

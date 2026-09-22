// Draws what the extractor decided. Cells carry their own confidence and reason
// codes, so the page never re-judges anything -- it only renders the verdict.

const $ = (id) => document.getElementById(id)
const statusEl = $('status')
const statsEl = $('stats')
const rejectedEl = $('rejected')
const tablesEl = $('tables')
const tip = $('tip')

const el = (tag, cls, text) => {
  const n = document.createElement(tag)
  if (cls) n.className = cls
  if (text !== undefined) n.textContent = text
  return n
}

function setStatus(text, isError = false) {
  statusEl.textContent = text
  statusEl.classList.toggle('err', isError)
}

async function loadSamples() {
  const r = await fetch('/api/samples')
  const { samples, max_pages } = await r.json()
  $('limits').textContent = `PDFs up to 8 MB. The first ${max_pages} pages are extracted here; the CLI has no such cap.`

  const box = $('samples')
  box.textContent = ''
  for (const s of samples) {
    const b = el('button')
    b.type = 'button'
    b.append(el('b', null, s.title), el('span', null, s.note))
    b.addEventListener('click', () => {
      ;[...box.querySelectorAll('button')].forEach((x) => x.classList.remove('active'))
      b.classList.add('active')
      extract({ sample: s.name }, s.title)
    })
    box.append(b)
  }
}

async function extract(payload, label) {
  setStatus(`Extracting ${label}… the pipeline runs per request, so this takes a few seconds.`)
  statsEl.hidden = true
  rejectedEl.hidden = true
  tablesEl.textContent = ''

  const body = new FormData()
  if (payload.sample) body.append('sample', payload.sample)
  if (payload.file) body.append('file', payload.file)

  let data
  try {
    const r = await fetch('/api/extract', { method: 'POST', body })
    data = await r.json()
    if (!r.ok || data.error) throw new Error(data.error || data.detail || `HTTP ${r.status}`)
  } catch (err) {
    setStatus(`Extraction failed: ${err.message}`, true)
    return
  }
  render(data, label)
}

function render(data, label) {
  const cells = data.tables.flatMap((t) => t.cells)
  const flagged = cells.filter((c) => c.reason_codes.length || c.mandatory_review)
  const pct = cells.length ? ((flagged.length / cells.length) * 100).toFixed(1) : '0.0'

  setStatus(
    `${label} — ${data.document.page_count ?? '?'} page(s) read, ` +
      `${(data.resources.wall_seconds ?? 0).toFixed(1)}s.`,
  )

  statsEl.textContent = ''
  const stats = [
    ['tables found', data.tables.length, ''],
    ['cells extracted', cells.length, ''],
    ['flagged for review', `${flagged.length} · ${pct}%`, 'warn'],
    ['regions refused', data.rejected_regions.length, 'good'],
  ]
  for (const [label2, value, cls] of stats) {
    const s = el('div', `stat ${cls}`.trim())
    s.append(el('b', null, String(value)), el('span', null, label2))
    statsEl.append(s)
  }
  statsEl.hidden = false

  // The refusals are the point: a contents page is geometrically a perfect table.
  if (data.rejected_regions.length) {
    rejectedEl.textContent = ''
    rejectedEl.append(el('h3', null, `${data.rejected_regions.length} region(s) refused`))
    rejectedEl.append(
      el('p', 'lead', 'Each of these is a well-formed grid that is not data. Extracting them would be a silent error.'),
    )
    const ul = el('ul')
    for (const r of data.rejected_regions.slice(0, 12)) {
      for (const j of r.rejections.slice(0, 2)) {
        const li = el('li')
        li.append(el('code', null, j.code || 'REJECTED'))
        li.append(document.createTextNode(` — page ${r.page_no}: ${j.message || ''}`))
        ul.append(li)
      }
    }
    rejectedEl.append(ul)
    rejectedEl.hidden = false
  }

  if (!data.tables.length) {
    tablesEl.append(
      el('p', 'muted', 'No tables extracted from this document — which, for the traps fixture, is the correct answer.'),
    )
    return
  }

  for (const t of data.tables) tablesEl.append(tableCard(t))
}

function tableCard(t) {
  const card = el('div', 'table-card')
  card.append(el('h3', null, t.title || t.table_id))

  const pages = t.start_page === t.end_page ? `page ${t.start_page}` : `pages ${t.start_page}–${t.end_page}`
  card.append(
    el(
      'div',
      'meta',
      `${t.table_id} · ${pages} · ${t.n_rows}×${t.n_cols} · ${t.regime} · confidence ${t.table_confidence ?? '—'} · ${t.n_flagged} flagged`,
    ),
  )

  // Lay the cells onto a dense grid so spans and blanks land where they belong.
  const grid = Array.from({ length: t.n_rows }, () => Array(t.n_cols).fill(null))
  for (const c of t.cells) {
    if (c.row < t.n_rows && c.col < t.n_cols) grid[c.row][c.col] = c
  }

  const wrap = el('div', 'scroll')
  const table = el('table')
  grid.forEach((row, ri) => {
    const tr = el('tr')
    row.forEach((c) => {
      const isHeader = c ? c.is_header : ri < t.n_header_rows
      const cell = el(isHeader ? 'th' : 'td')
      if (!c) {
        cell.textContent = ''
      } else {
        cell.textContent = c.raw_text || ''
        const codes = c.reason_codes || []
        if (c.mandatory_review) cell.classList.add('must')
        else if (codes.length) cell.classList.add('flag')
        if (codes.length) {
          cell.append(el('span', 'badge', '⚑'))
          cell.addEventListener('mouseenter', (e) => showTip(e, c))
          cell.addEventListener('mouseleave', hideTip)
        }
      }
      tr.append(cell)
    })
    table.append(tr)
  })
  wrap.append(table)
  card.append(wrap)

  const legend = el('div', 'legend')
  legend.innerHTML =
    '<span><i style="background:rgba(251,191,36,0.35)"></i>flagged for review</span>' +
    '<span><i style="background:rgba(248,113,113,0.4)"></i>mandatory review</span>' +
    '<span>hover a flagged cell for its reason codes</span>'
  card.append(legend)

  if (t.unit_scale && t.unit_scale.note) {
    card.append(
      el('p', 'note', `Scale: ${t.unit_scale.note} (read from ${t.unit_scale.source}, page ${t.unit_scale.page_no}).`),
    )
  }
  if (t.reconciliation && Object.keys(t.reconciliation).length) {
    const rec = el('p', 'note')
    rec.append(document.createTextNode('Reconciliation: '))
    rec.append(el('code', null, JSON.stringify(t.reconciliation).slice(0, 220)))
    card.append(rec)
  }
  for (const f of (t.footnotes || []).slice(0, 4)) {
    card.append(el('p', 'note', `Footnote ${f.marker} (page ${f.page_no}): ${f.text}`))
  }
  return card
}

function showTip(e, c) {
  tip.textContent = ''
  tip.append(el('b', null, c.reason_codes.join(' · ')))
  const bits = []
  if (c.confidence !== null && c.confidence !== undefined) bits.push(`confidence ${c.confidence}`)
  if (c.normalized_value !== null && c.normalized_value !== undefined) bits.push(`normalised ${c.normalized_value}`)
  if (c.value_type) bits.push(c.value_type)
  if (c.mandatory_review) bits.push('mandatory review')
  tip.append(document.createTextNode(bits.join(' · ')))
  tip.hidden = false
  const r = e.target.getBoundingClientRect()
  tip.style.left = `${Math.min(r.left, window.innerWidth - 340)}px`
  tip.style.top = `${r.bottom + 6}px`
}
const hideTip = () => { tip.hidden = true }

$('file').addEventListener('change', (e) => {
  const f = e.target.files[0]
  if (f) extract({ file: f }, f.name)
})

loadSamples().catch(() => setStatus('Could not reach the API.', true))

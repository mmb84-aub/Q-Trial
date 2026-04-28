/**
 * Q-Trial Pipeline Tracker & Static Report Viewer
 * Standalone JS — no framework dependencies.
 */

/* ── Pipeline Tracker ────────────────────────────────────────────────── */

function renderPipeline(stages) {
  var container = document.getElementById('pipeline-stages');
  if (!container) return;
  container.innerHTML = '';

  for (var i = 0; i < stages.length; i++) {
    var s = stages[i];
    var stateClass = s.state || 'pending'; // done | active | pending

    // Node
    var node = document.createElement('div');
    node.className = 'stage-node';

    var icon = document.createElement('div');
    icon.className = 'stage-icon ' + stateClass;
    if (stateClass === 'done') {
      icon.innerHTML = '&#10003;';
    } else if (stateClass === 'active') {
      icon.innerHTML = '&#9679;';
    } else {
      icon.textContent = String(i + 1);
    }

    var label = document.createElement('div');
    label.className = 'stage-label ' + stateClass;
    label.textContent = s.label;

    node.appendChild(icon);
    node.appendChild(label);

    var stage = document.createElement('div');
    stage.className = 'stage';
    stage.appendChild(node);

    // Connector (except after last)
    if (i < stages.length - 1) {
      var conn = document.createElement('div');
      conn.className = 'stage-connector' + (stateClass === 'done' ? ' done' : '');
      stage.appendChild(conn);
    }

    container.appendChild(stage);
  }
}


/* ── Static Report Viewer ────────────────────────────────────────────── */

function renderStaticReport(markdownText, containerId) {
  var container = document.getElementById(containerId);
  if (!container || !markdownText) return;

  // Split markdown into sections by # headings
  var lines = markdownText.split('\n');
  var sections = [];
  var current = null;

  for (var i = 0; i < lines.length; i++) {
    var line = lines[i];
    // Match ## or ### section headers
    var match = line.match(/^(#{1,3})\s+(.+)/);
    if (match && match[1].length <= 2) {
      if (current) sections.push(current);
      current = { title: match[2].trim(), content: [] };
    } else if (current) {
      current.content.push(line);
    }
  }
  if (current) sections.push(current);

  // Skip the first section if it's just the report title
  if (sections.length > 0 && sections[0].title.indexOf('Static Analysis Report') >= 0) {
    sections.shift();
  }

  var html = '<div class="report-sections">';

  for (var j = 0; j < sections.length; j++) {
    var sec = sections[j];
    var bodyHtml = markdownToHtml(sec.content.join('\n'));
    var secId = 'sec-' + j;

    html += '<div class="report-section">';
    html += '<div class="report-section-header" onclick="toggleSection(\'' + secId + '\')">';
    html += '<div class="report-section-title">';
    html += '<span class="section-num">' + (j + 1) + '</span>';
    html += escapeHtml(sec.title);
    html += '</div>';
    html += '<span class="report-section-chevron">&#9660;</span>';
    html += '</div>';
    html += '<div class="report-section-body" id="' + secId + '">';
    html += bodyHtml;
    html += '</div></div>';
  }

  html += '</div>';

  // Raw markdown toggle
  html += '<div class="report-raw-toggle">';
  html += '<button class="report-raw-btn" onclick="toggleRawReport()">View Raw Markdown</button>';
  html += '</div>';
  html += '<div class="report-raw-content" id="raw-report">' + escapeHtml(markdownText) + '</div>';

  container.innerHTML = html;
}

function toggleSection(id) {
  var body = document.getElementById(id);
  if (!body) return;
  var header = body.previousElementSibling;
  body.classList.toggle('open');
  if (header) header.classList.toggle('open');
}

function toggleRawReport() {
  var el = document.getElementById('raw-report');
  if (el) el.classList.toggle('open');
}


/* ── Minimal Markdown → HTML ─────────────────────────────────────────── */

function escapeHtml(text) {
  var div = document.createElement('div');
  div.appendChild(document.createTextNode(text));
  return div.innerHTML;
}

function markdownToHtml(md) {
  if (!md) return '';
  var lines = md.split('\n');
  var html = '';
  var inTable = false;
  var inList = false;
  var inBlockquote = false;

  for (var i = 0; i < lines.length; i++) {
    var line = lines[i];
    var trimmed = line.trim();

    // Skip empty lines
    if (trimmed === '') {
      if (inList) { html += '</ul>'; inList = false; }
      if (inBlockquote) { html += '</blockquote>'; inBlockquote = false; }
      continue;
    }

    // Table separator row (skip)
    if (/^\|[\s\-:|]+\|$/.test(trimmed)) {
      continue;
    }

    // Table row
    if (trimmed.charAt(0) === '|' && trimmed.charAt(trimmed.length - 1) === '|') {
      var cells = trimmed.slice(1, -1).split('|').map(function(c) { return c.trim(); });
      if (!inTable) {
        html += '<table><thead><tr>';
        for (var c = 0; c < cells.length; c++) {
          html += '<th>' + inlineMarkdown(cells[c]) + '</th>';
        }
        html += '</tr></thead><tbody>';
        inTable = true;
      } else {
        html += '<tr>';
        for (var c2 = 0; c2 < cells.length; c2++) {
          html += '<td>' + inlineMarkdown(cells[c2]) + '</td>';
        }
        html += '</tr>';
      }
      continue;
    } else if (inTable) {
      html += '</tbody></table>';
      inTable = false;
    }

    // Heading
    var hMatch = trimmed.match(/^(#{1,4})\s+(.+)/);
    if (hMatch) {
      var level = Math.min(hMatch[1].length + 1, 6);
      html += '<h' + level + '>' + inlineMarkdown(hMatch[2]) + '</h' + level + '>';
      continue;
    }

    // Blockquote
    if (trimmed.charAt(0) === '>') {
      var bqText = trimmed.slice(1).trim();
      if (!inBlockquote) { html += '<blockquote>'; inBlockquote = true; }
      html += '<p>' + inlineMarkdown(bqText) + '</p>';
      continue;
    }

    // List items
    if (/^[-*]\s/.test(trimmed)) {
      if (!inList) { html += '<ul>'; inList = true; }
      html += '<li>' + inlineMarkdown(trimmed.slice(2)) + '</li>';
      continue;
    }

    // Horizontal rule
    if (/^---+$/.test(trimmed)) {
      html += '<hr>';
      continue;
    }

    // Paragraph
    html += '<p>' + inlineMarkdown(trimmed) + '</p>';
  }

  if (inTable) html += '</tbody></table>';
  if (inList) html += '</ul>';
  if (inBlockquote) html += '</blockquote>';

  return html;
}

function inlineMarkdown(text) {
  text = escapeHtml(text);
  // Bold
  text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic
  text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Inline code
  text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
  return text;
}

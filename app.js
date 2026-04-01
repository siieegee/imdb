/* =============================================================
   DATA — mirrors your ML pipeline output.
   Each movie has the exact features your model trained on:
     runtimeMinutes, startYear, numVotes, averageRating
   plus a clusterId assigned by K-Means.

   TO USE YOUR REAL DATA: replace the MOVIES array below with
   a JSON export from your pipeline. Each object needs:
     { id, title, year, runtime, votes, rating, cluster, emoji }
   Run export_data.py to generate this from your IMDB CSVs.
============================================================= */

const CLUSTERS = [
  { id: 0, name: "Short cult classics",       color: "#7b6bb5", dim: "rgba(123,107,181,0.12)", dot: "#7b6bb5" },
  { id: 1, name: "Epic prestige dramas",      color: "#c9a84c", dim: "rgba(201,168,76,0.12)",  dot: "#c9a84c" },
  { id: 2, name: "Mainstream crowd-pleasers", color: "#4a9e8a", dim: "rgba(74,158,138,0.12)",  dot: "#4a9e8a" },
  { id: 3, name: "Arthouse & indie",          color: "#c46a4a", dim: "rgba(196,106,74,0.12)",  dot: "#c46a4a" },
  { id: 4, name: "Blockbuster spectacles",    color: "#9e6fb5", dim: "rgba(158,111,181,0.12)", dot: "#9e6fb5" },
];

const MOVIES = [
  // Cluster 0 — Short cult classics
  { id:1,  title:"Pulp Fiction",             year:1994, runtime:154, votes:2100000, rating:8.9, cluster:0, emoji:"🎲" },
  { id:2,  title:"Reservoir Dogs",           year:1992, runtime:99,  votes:1100000, rating:8.3, cluster:0, emoji:"💼" },
  { id:3,  title:"Fargo",                    year:1996, runtime:98,  votes:760000,  rating:8.1, cluster:0, emoji:"❄️" },
  { id:4,  title:"The Big Lebowski",         year:1998, runtime:117, votes:760000,  rating:8.1, cluster:0, emoji:"🎳" },
  { id:5,  title:"A Clockwork Orange",       year:1971, runtime:136, votes:850000,  rating:8.3, cluster:0, emoji:"🎩" },
  { id:6,  title:"Trainspotting",            year:1996, runtime:93,  votes:720000,  rating:8.1, cluster:0, emoji:"💊" },
  { id:7,  title:"Se7en",                    year:1995, runtime:127, votes:1700000, rating:8.6, cluster:0, emoji:"🔎" },
  { id:8,  title:"Fight Club",               year:1999, runtime:139, votes:2200000, rating:8.8, cluster:0, emoji:"👊" },
  // Cluster 1 — Epic prestige dramas
  { id:9,  title:"The Shawshank Redemption", year:1994, runtime:142, votes:2700000, rating:9.3, cluster:1, emoji:"🔒" },
  { id:10, title:"Schindler's List",         year:1993, runtime:195, votes:1400000, rating:9.0, cluster:1, emoji:"📜" },
  { id:11, title:"The Green Mile",           year:1999, runtime:189, votes:1100000, rating:8.6, cluster:1, emoji:"🌿" },
  { id:12, title:"Goodfellas",               year:1990, runtime:146, votes:1200000, rating:8.7, cluster:1, emoji:"🍷" },
  { id:13, title:"The Godfather",            year:1972, runtime:175, votes:1900000, rating:9.2, cluster:1, emoji:"🌹" },
  { id:14, title:"Forrest Gump",             year:1994, runtime:142, votes:2000000, rating:8.8, cluster:1, emoji:"🍫" },
  { id:15, title:"Saving Private Ryan",      year:1998, runtime:169, votes:1300000, rating:8.6, cluster:1, emoji:"🪖" },
  { id:16, title:"Braveheart",               year:1995, runtime:178, votes:1100000, rating:8.3, cluster:1, emoji:"⚔️" },
  // Cluster 2 — Mainstream crowd-pleasers
  { id:17, title:"The Lion King",            year:1994, runtime:88,  votes:1000000, rating:8.5, cluster:2, emoji:"🦁" },
  { id:18, title:"Home Alone",               year:1990, runtime:103, votes:750000,  rating:7.7, cluster:2, emoji:"🏠" },
  { id:19, title:"Toy Story",                year:1995, runtime:81,  votes:1000000, rating:8.3, cluster:2, emoji:"🪀" },
  { id:20, title:"Jumanji",                  year:1995, runtime:104, votes:450000,  rating:7.0, cluster:2, emoji:"🎲" },
  { id:21, title:"Titanic",                  year:1997, runtime:194, votes:1200000, rating:7.9, cluster:2, emoji:"🚢" },
  { id:22, title:"The Matrix",               year:1999, runtime:136, votes:1900000, rating:8.7, cluster:2, emoji:"💊" },
  { id:23, title:"Gladiator",                year:2000, runtime:155, votes:1500000, rating:8.5, cluster:2, emoji:"🏛️" },
  { id:24, title:"Pirates of Caribbean",     year:2003, runtime:143, votes:1100000, rating:8.0, cluster:2, emoji:"⚓" },
  // Cluster 3 — Arthouse & indie
  { id:25, title:"Mulholland Drive",         year:2001, runtime:147, votes:440000,  rating:7.9, cluster:3, emoji:"🌃" },
  { id:26, title:"Lost in Translation",      year:2003, runtime:102, votes:500000,  rating:7.8, cluster:3, emoji:"🗼" },
  { id:27, title:"Eternal Sunshine",         year:2004, runtime:108, votes:890000,  rating:8.3, cluster:3, emoji:"☀️" },
  { id:28, title:"Her",                      year:2013, runtime:126, votes:670000,  rating:8.0, cluster:3, emoji:"📱" },
  { id:29, title:"Moonlight",                year:2016, runtime:111, votes:380000,  rating:7.4, cluster:3, emoji:"🌙" },
  { id:30, title:"Parasite",                 year:2019, runtime:132, votes:850000,  rating:8.5, cluster:3, emoji:"🏚️" },
  { id:31, title:"The Favourite",            year:2018, runtime:119, votes:280000,  rating:7.5, cluster:3, emoji:"👑" },
  { id:32, title:"A Ghost Story",            year:2017, runtime:92,  votes:110000,  rating:6.9, cluster:3, emoji:"👻" },
  // Cluster 4 — Blockbuster spectacles
  { id:33, title:"Inception",                year:2010, runtime:148, votes:2300000, rating:8.8, cluster:4, emoji:"🌀" },
  { id:34, title:"The Dark Knight",          year:2008, runtime:152, votes:2800000, rating:9.0, cluster:4, emoji:"🦇" },
  { id:35, title:"Interstellar",             year:2014, runtime:169, votes:1900000, rating:8.6, cluster:4, emoji:"🪐" },
  { id:36, title:"Avengers: Endgame",        year:2019, runtime:181, votes:1200000, rating:8.4, cluster:4, emoji:"⚡" },
  { id:37, title:"Avatar",                   year:2009, runtime:162, votes:1300000, rating:7.9, cluster:4, emoji:"🌿" },
  { id:38, title:"The Prestige",             year:2006, runtime:130, votes:1300000, rating:8.5, cluster:4, emoji:"🎩" },
  { id:39, title:"Dune",                     year:2021, runtime:155, votes:800000,  rating:8.0, cluster:4, emoji:"🏜️" },
  { id:40, title:"Mad Max: Fury Road",       year:2015, runtime:120, votes:1000000, rating:8.1, cluster:4, emoji:"🚗" },
];

/* ── HELPERS ── */

function similarity(a, b) {
  const norm = (v, min, max) => (v - min) / (max - min || 1);
  const dist = Math.sqrt(
    Math.pow(norm(b.runtime, 80, 200)    - norm(a.runtime, 80, 200),    2) * 1.5 +
    Math.pow(norm(b.year,    1970, 2024) - norm(a.year,    1970, 2024), 2) * 1.2 +
    Math.pow(norm(b.votes,   0, 3000000) - norm(a.votes,   0, 3000000), 2) * 1.8 +
    Math.pow(norm(b.rating,  5, 10)      - norm(a.rating,  5, 10),      2) * 1.0
  );
  const clusterBonus = a.cluster === b.cluster ? 0.15 : 0;
  return Math.min(99, Math.round((1 - dist / 2.5 + clusterBonus) * 100));
}

function fmt(n) {
  if (n >= 1000000) return (n / 1000000).toFixed(1) + "M";
  if (n >= 1000)    return Math.round(n / 1000) + "K";
  return String(n);
}

function clusterOf(id) {
  return CLUSTERS.find(c => c.id === id);
}

/* ── STATE ── */
let activeCluster     = null;
let selectedMovie     = null;
let currentSort       = "match";
let dropdownFocusIdx  = -1;   // keyboard nav index in dropdown
let dropdownItems     = [];   // current dropdown movie hits

/* ── CLUSTER CHIPS ── */
function renderClusters() {
  document.getElementById("clustersRow").innerHTML = CLUSTERS.map(c => `
    <button
      class="cluster-chip ${activeCluster === c.id ? "active" : ""}"
      onclick="browseCluster(${c.id})"
      aria-pressed="${activeCluster === c.id}"
      aria-label="Browse ${c.name} cluster"
    >
      <span class="chip-dot" style="background:${c.dot}" aria-hidden="true"></span>
      ${c.name}
    </button>
  `).join("");
}

function browseCluster(id) {
  activeCluster = id;
  renderClusters();
  const first = MOVIES.find(m => m.cluster === id);
  if (first) showMovie(first.id);
}

/* ── SEARCH with keyboard navigation ── */
const searchInput = document.getElementById("searchInput");
const dropdown    = document.getElementById("dropdown");

function closeDropdown() {
  dropdown.classList.remove("open");
  dropdown.innerHTML = "";
  dropdownItems    = [];
  dropdownFocusIdx = -1;
  searchInput.setAttribute("aria-expanded", "false");
}

function openDropdown(hits) {
  dropdownItems    = hits;
  dropdownFocusIdx = -1;

  dropdown.innerHTML = hits.map((m, i) => {
    const cl = clusterOf(m.cluster);
    return `
      <div
        class="dropdown-item"
        role="option"
        id="dd-item-${i}"
        aria-selected="false"
        tabindex="-1"
        data-id="${m.id}"
        onclick="pickMovie(${m.id})"
      >
        <div class="di-icon" style="background:${cl.dim}" aria-hidden="true">${m.emoji}</div>
        <div>
          <p class="di-title">${m.title}</p>
          <p class="di-meta">${m.year} · ${m.runtime} min · <span aria-label="Rating ${m.rating} out of 10">★ ${m.rating}</span></p>
        </div>
      </div>`;
  }).join("");

  dropdown.classList.add("open");
  searchInput.setAttribute("aria-expanded", "true");
  searchInput.setAttribute("aria-activedescendant", "");
}

function setDropdownFocus(idx) {
  const items = dropdown.querySelectorAll(".dropdown-item");
  items.forEach((el, i) => {
    el.setAttribute("aria-selected", i === idx ? "true" : "false");
    if (i === idx) el.classList.add("focused");
    else           el.classList.remove("focused");
  });
  if (idx >= 0 && items[idx]) {
    searchInput.setAttribute("aria-activedescendant", `dd-item-${idx}`);
    items[idx].scrollIntoView({ block: "nearest" });
  } else {
    searchInput.setAttribute("aria-activedescendant", "");
  }
  dropdownFocusIdx = idx;
}

searchInput.addEventListener("input", function () {
  const q = this.value.trim().toLowerCase();
  if (q.length < 2) { closeDropdown(); return; }

  const hits = MOVIES.filter(m => m.title.toLowerCase().includes(q)).slice(0, 6);
  if (!hits.length) { closeDropdown(); return; }

  openDropdown(hits);
});

searchInput.addEventListener("keydown", function (e) {
  const items = dropdown.querySelectorAll(".dropdown-item");
  const total = items.length;

  if (e.key === "ArrowDown") {
    e.preventDefault();
    if (!dropdown.classList.contains("open")) return;
    setDropdownFocus((dropdownFocusIdx + 1) % total);
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    if (!dropdown.classList.contains("open")) return;
    setDropdownFocus(dropdownFocusIdx <= 0 ? total - 1 : dropdownFocusIdx - 1);
  } else if (e.key === "Enter") {
    e.preventDefault();
    if (dropdownFocusIdx >= 0 && dropdownItems[dropdownFocusIdx]) {
      pickMovie(dropdownItems[dropdownFocusIdx].id);
    } else if (dropdownItems.length === 1) {
      pickMovie(dropdownItems[0].id);
    }
  } else if (e.key === "Escape") {
    closeDropdown();
    searchInput.focus();
  }
});

// Close dropdown when clicking outside
document.addEventListener("click", e => {
  if (!e.target.closest(".search-wrap")) closeDropdown();
});

// Also close on focus-out of the whole search widget
document.getElementById("searchInput").addEventListener("blur", function (e) {
  // Small delay so click on a dropdown item registers first
  setTimeout(() => {
    if (!document.activeElement.closest(".search-wrap")) closeDropdown();
  }, 150);
});

function pickMovie(id) {
  closeDropdown();
  searchInput.value = "";
  showMovie(id);
}

/* ── SHOW MOVIE ── */
function showMovie(id) {
  selectedMovie = MOVIES.find(m => m.id === id);
  if (!selectedMovie) return;

  activeCluster = selectedMovie.cluster;
  renderClusters();
  renderSelectedPanel();
  renderWhyPanel();
  renderResults();

  const mainSection = document.getElementById("mainSection");
  mainSection.classList.add("visible");
  document.getElementById("heroSection").style.paddingBottom = "1.5rem";

  // Smooth scroll then move focus to results heading for screen readers
  mainSection.scrollIntoView({ behavior: "smooth", block: "start" });
  setTimeout(() => {
    const heading = document.getElementById("resultsTitle");
    if (heading) { heading.setAttribute("tabindex", "-1"); heading.focus({ preventScroll: true }); }
  }, 400);
}

function renderSelectedPanel() {
  const m  = selectedMovie;
  const cl = clusterOf(m.cluster);
  document.getElementById("selectedPanel").innerHTML = `
    <div class="sel-poster" style="background:${cl.dim}" aria-hidden="true">${m.emoji}</div>
    <div>
      <p class="sel-title">${m.title}</p>
      <div class="sel-meta" aria-label="${m.title} details: ${m.year}, ${m.runtime} minutes, ${fmt(m.votes)} votes">
        <span class="sel-tag">${m.year}</span>
        <span class="sel-tag">${m.runtime} min</span>
        <span class="sel-tag">${fmt(m.votes)} votes</span>
      </div>
      <div class="sel-cluster">
        <span style="font-size:12px;color:var(--text3)" aria-hidden="true">Cluster —</span>
        <span
          class="sel-cluster-badge"
          style="background:${cl.dim};color:${cl.color};border:0.5px solid ${cl.color}44"
          aria-label="Cluster: ${cl.name}"
        >
          ${cl.name}
        </span>
      </div>
    </div>
    <div class="sel-stats" aria-label="Rating ${m.rating} out of 10, ${fmt(m.votes)} votes">
      <div class="sel-rating" aria-hidden="true">★ ${m.rating}</div>
      <div class="sel-votes">${fmt(m.votes)} votes</div>
    </div>
  `;
}

function renderWhyPanel() {
  const m  = selectedMovie;
  const cl = clusterOf(m.cluster);
  const group = MOVIES.filter(x => x.cluster === m.cluster);

  const avgRuntime = Math.round(group.reduce((a, x) => a + x.runtime, 0) / group.length);
  const avgVotes   = Math.round(group.reduce((a, x) => a + x.votes,   0) / group.length);
  const avgRating  = (group.reduce((a, x) => a + x.rating, 0) / group.length).toFixed(1);

  const runtimePct = Math.round((m.runtime / 210)           * 100);
  const votesPct   = Math.round((m.votes   / 3000000)       * 100);
  const ratingPct  = Math.round(((m.rating - 5) / 5)        * 100);
  const yearPct    = Math.round(((m.year   - 1970) / 55)    * 100);
  const era        = m.year < 1990 ? "classic" : m.year < 2005 ? "90s–2000s" : "modern";

  const factor = (label, value, pct, avg, avgLabel) => `
    <div class="factor" role="group" aria-label="${label}: ${value}. Cluster average: ${avgLabel}">
      <p class="factor-label" aria-hidden="true">${label}</p>
      <p class="factor-value" aria-hidden="true">${value}</p>
      <div class="factor-bar" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100" aria-label="${label} ${pct}% of maximum">
        <div class="factor-bar-fill" style="width:${pct}%;background:${cl.color}"></div>
      </div>
      <p style="font-size:10px;color:var(--text3);margin-top:4px" aria-hidden="true">${avg}</p>
    </div>`;

  document.getElementById("whyPanel").innerHTML = `
    <p class="why-title" id="why-title">Why it's in this cluster — feature profile</p>
    <div class="why-factors" aria-labelledby="why-title">
      ${factor("Runtime",           `${m.runtime} min`, runtimePct, `Cluster avg: ${avgRuntime} min`, `${avgRuntime} minutes`)}
      ${factor("Popularity (votes)",fmt(m.votes),        votesPct,   `Cluster avg: ${fmt(avgVotes)}`,  `${fmt(avgVotes)} votes`)}
      ${factor("Rating",            `★ ${m.rating}`,    ratingPct,  `Cluster avg: ★ ${avgRating}`,    `${avgRating} out of 10`)}
      ${factor("Release year",      `${m.year}`,         yearPct,    `Era: ${era}`,                    `${era} era`)}
    </div>
  `;
}

function sortCards(type, btn) {
  currentSort = type;
  document.querySelectorAll(".sort-btn").forEach(b => {
    b.classList.remove("active");
    b.setAttribute("aria-pressed", "false");
  });
  btn.classList.add("active");
  btn.setAttribute("aria-pressed", "true");
  renderResults();
}

function renderResults() {
  const m  = selectedMovie;
  let candidates = MOVIES
    .filter(x => x.id !== m.id)
    .map(x => ({ ...x, matchScore: similarity(m, x) }))
    .filter(x => x.matchScore > 50);

  if (currentSort === "match")  candidates.sort((a, b) => b.matchScore - a.matchScore);
  if (currentSort === "rating") candidates.sort((a, b) => b.rating     - a.rating);
  if (currentSort === "year")   candidates.sort((a, b) => b.year       - a.year);
  if (currentSort === "votes")  candidates.sort((a, b) => b.votes      - a.votes);

  candidates = candidates.slice(0, 12);

  document.getElementById("resultsTitle").textContent = `Similar to ${m.title}`;
  document.getElementById("resultsCount").textContent = `${candidates.length} titles found`;

  if (!candidates.length) {
    document.getElementById("moviesGrid").innerHTML = `
      <div class="empty" style="grid-column:1/-1" role="status">
        <div class="empty-icon" aria-hidden="true">🎬</div>
        <p>No similar titles found.</p>
      </div>`;
    return;
  }

  const clusterBg = { 0:"#13101e", 1:"#1a1610", 2:"#101a18", 3:"#1a1210", 4:"#141020" };

  document.getElementById("moviesGrid").innerHTML = candidates.map((c, i) => {
    const ccl         = clusterOf(c.cluster);
    const sameCluster = c.cluster === m.cluster;
    const matchColor  = c.matchScore >= 85 ? "#4a9e8a" : c.matchScore >= 70 ? "#c9a84c" : "#6b6b75";
    const matchLabel  = c.matchScore >= 85 ? "High match" : c.matchScore >= 70 ? "Good match" : "Moderate match";
    const bgColor     = clusterBg[c.cluster] || "#141414";

    return `
      <div
        class="movie-card"
        style="animation-delay:${i * 0.04}s"
        role="listitem"
        tabindex="0"
        aria-label="${c.title}, ${c.year}, ${c.runtime} minutes, rated ${c.rating} out of 10, ${c.matchScore}% match — ${matchLabel}. ${sameCluster ? 'Same cluster as selected movie.' : 'Cluster ' + c.cluster + '.'}"
        onclick="showMovie(${c.id})"
        onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();showMovie(${c.id})}"
      >
        <div class="card-top" style="background:${bgColor}">
          <span aria-hidden="true">${c.emoji}</span>
          <div class="match-strip" aria-hidden="true">
            <div class="match-fill" style="width:${c.matchScore}%;background:${matchColor}"></div>
          </div>
        </div>
        <div class="card-body">
          <p class="card-title" title="${c.title}">${c.title}</p>
          <p class="card-year" aria-hidden="true">${c.year} · ${c.runtime} min</p>
          <div class="card-row" aria-hidden="true">
            <span class="card-rating">★ ${c.rating}</span>
            <span class="card-match" style="color:${matchColor}">${c.matchScore}% match</span>
          </div>
          <div class="card-badges" aria-hidden="true">
            ${sameCluster
              ? `<span class="badge" style="background:${ccl.dim};color:${ccl.color}">same cluster</span>`
              : `<span class="badge" style="background:var(--bg4);color:var(--text3)">cluster ${c.cluster}</span>`
            }
            <span class="badge" style="background:var(--bg4);color:var(--text3)">${fmt(c.votes)} votes</span>
          </div>
        </div>
      </div>`;
  }).join("");
}

/* ── INIT ── */
renderClusters();

/* ── PAGE NAVIGATION ── */
function showSection(name) {
  // Hide all sections
  document.querySelectorAll('.page-section').forEach(el => {
    el.classList.remove('active');
    el.hidden = true;
  });

  // Show target
  const target = document.getElementById(name + 'Section');
  if (target) {
    target.classList.add('active');
    target.hidden = false;
    // Move focus to section heading for screen readers
    const heading = target.querySelector('h1');
    if (heading) {
      heading.setAttribute('tabindex', '-1');
      heading.focus({ preventScroll: true });
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // Update nav button states
  document.querySelectorAll('.nav-link').forEach(btn => {
    const isActive = btn.dataset.section === name;
    btn.classList.toggle('active', isActive);
    btn.setAttribute('aria-current', isActive ? 'page' : 'false');
  });
}
/* ─────────────────────────────────────────────
   SonicLens — Audio Similarity Explorer
   ───────────────────────────────────────────── */

const STEM_COLORS = {
  bass:'#f97316', drums:'#ef4444', guitar:'#eab308',
  other:'#8b5cf6', piano:'#06b6d4', vocals:'#ec4899', full:'#7c5cfc'
};
const ALL_STEMS = ['bass','drums','guitar','other','piano','vocals','full'];
const PLAYABLE = ['bass','drums','guitar','other','piano','vocals'];

const BACKEND_URL = 'https://mynkpdr--soniclens-backend-web.modal.run'; // Hugging Face Backend

let DATA = null, currentStem = 'full', selectedSongIdx = null;
let clusterThresholdFromURL = false;
let clusterSliderDefaulted = false;
let customClusterMode = false;
let customClusterWeights = Object.fromEntries(PLAYABLE.map(stem => [stem, 1]));
let customClusterSimilarityMatrix = null;
const MAX_COMPARE_SONGS = 5;
let compareSelection = [];
let compareFocusedPair = null;
let compareSelectionNotice = '';
let compareMatrixStem = 'full';

// Utility: Debounce wrapper for rapid UI events
function debounce(func, wait) {
  let timeout;
  return function(...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
}

/* ── Waveform state ── */
let audioCtx, audioBuffer, waveformData = [];
let currentPlayingFile = '', currentPlayingStem = '';

/* ═══════════════════════════════════════════
   BOOTSTRAP
   ═══════════════════════════════════════════ */
async function init() {
  clusterThresholdFromURL = false;
  clusterSliderDefaulted = false;
  const artistsRes = await fetch('./data/artists.json');
  const artists = await artistsRes.json();
  const selector = document.getElementById('artist-selector');
  
  const localSongs = await getAllAddedSongs();
  const customArtists = [...new Set(localSongs.map(s => s.songObj.singer))];
  
  customArtists.forEach(ca => {
      if (!artists.find(a => a.name === ca)) {
          artists.push({ id: ca, name: ca });
      }
  });

  // Populate dropdown
  selector.innerHTML = '';
  artists.forEach(a => {
    const opt = document.createElement('option');
    opt.value = a.id;
    opt.textContent = a.name;
    selector.appendChild(opt);
  });

  // Populate upload tab singer dropdown
  const uploadSinger = document.getElementById('yt-singer');
  if (uploadSinger) {
    uploadSinger.innerHTML = '';
    artists.forEach(a => {
      const opt = document.createElement('option');
      opt.value = a.id;
      opt.textContent = a.name;
      uploadSinger.appendChild(opt);
    });
  }

  // Get active artist from URL or fallback
  const urlParams = new URLSearchParams(window.location.search);
  let activeArtist = urlParams.get('artist');
  if (!activeArtist || !artists.some(a => a.id === activeArtist)) {
    activeArtist = artists[0].id; // Fallback
  }
  selector.value = activeArtist;
  if (uploadSinger) uploadSinger.value = activeArtist;
  
  // Handle change
  selector.addEventListener('change', (e) => {
    const newUrl = new URL(window.location.href);
    newUrl.searchParams.set('artist', e.target.value);
    
    // Clear out song selection and dependent state before reloading
    newUrl.searchParams.delete('song');
    window.location.href = newUrl.toString();
  });

  try {
    DATA = await (await fetch(`./data/${activeArtist}.json`)).json();
  } catch(e) {
    DATA = { songs: [] };
  }

  // Ensure mandatory fields exist for cleaned JSONs
  if (!DATA.songs) DATA.songs = [];
  if (!DATA.similarities) DATA.similarities = Object.fromEntries(ALL_STEMS.map(s => [s, []]));
  if (!DATA.tsne) DATA.tsne = Object.fromEntries(ALL_STEMS.map(s => [s, []]));
  if (!DATA.stem_energy) DATA.stem_energy = Object.fromEntries(ALL_STEMS.map(s => [s, []]));
  if (!DATA.clusters) DATA.clusters = {};

  await fetchExistingEmbeddings(activeArtist);

  const artistLocalSongs = localSongs.filter(s => s.songObj.singer === activeArtist || s.songObj.artist_group === activeArtist);
  artistLocalSongs.forEach(item => {
      const idx = DATA.songs.length;
      const sObj = { ...item.songObj, index: idx };
      DATA.songs.push(sObj);
      
      ALL_STEMS.forEach(stem => {
          if (!DATA.stem_energy[stem]) DATA.stem_energy[stem] = [];
          DATA.stem_energy[stem].push(1.0);
          
          if (!DATA.similarities[stem]) DATA.similarities[stem] = [];
          
          // Ensure similarity row is correctly sized
          const simRow = item.similarities && item.similarities[stem] ? [...item.similarities[stem]] : new Array(idx + 1).fill(0);
          while (simRow.length <= idx) simRow.push(0);
          simRow[idx] = 1.0;
          DATA.similarities[stem].push(simRow);
          
          for (let i = 0; i < idx; i++) {
              if (!DATA.similarities[stem][i]) DATA.similarities[stem][i] = new Array(idx).fill(0);
              while (DATA.similarities[stem][i].length < idx) DATA.similarities[stem][i].push(0);
              DATA.similarities[stem][i].push(simRow[i] || 0);
          }
          
          if (!DATA.tsne[stem]) DATA.tsne[stem] = [];
          DATA.tsne[stem].push([0.0, 0.0]);
          
          if (cachedEmbeddings && cachedEmbeddings[stem]) {
              if (item.embeddings && item.embeddings[stem]) {
                  cachedEmbeddings[stem].push(new Float32Array(item.embeddings[stem]));
              } else {
                  // Fallback to zeros if missing to keep indices aligned
                  cachedEmbeddings[stem].push(new Float32Array(3072).fill(0));
              }
          }
      });
  });

  document.getElementById('loader').style.display = 'none';

  setupTabs(); buildAllStemBars(); setupModal(); setupPlayerControls();

  // Apply URL state AFTER components are built so chips/panels exist in DOM
  parseURLState();

  // Ensure a panel is always visible
  if (!document.querySelector('.panel.active')) {
    document.querySelector('.tab-btn[data-tab="cluster"]').classList.add('active');
    document.getElementById('panel-cluster').classList.add('active');
  }

  // Re-sync all stem chips to currentStem after URL parse
  document.querySelectorAll('.stem-chip').forEach(c =>
    c.classList.toggle('active', c.dataset.stem === currentStem)
  );


  renderSimilarityList();
  buildCompareSelects();

  setupUploadTab();

  // Render heavy tabs if directly linked
  const activeTab = document.querySelector('.tab-btn.active')?.dataset.tab;
  if (activeTab === 'heatmap') renderHeatmap();
  if (activeTab === 'cluster') renderCluster();

  if (selectedSongIdx !== null) {
    highlightSidebarItem(selectedSongIdx);
    renderSimilarityResults(selectedSongIdx);
  }

  d3.select(window).on('resize.heatmap', renderHeatmap);
}


/* ═══════════════════════════════════════════
   UPLOAD TAB & NPY UTILS
   ═══════════════════════════════════════════ */

/**
 * Shows/hides a pipeline step card.
 * @param {string} id - Element id, e.g. 'step-2'
 * @param {boolean} visible
 */
function showStep(id, visible = true) {
  const el = document.getElementById(id);
  if (el) el.style.display = visible ? 'block' : 'none';
}

/**
 * Updates a status element with a message, applying an error style when needed.
 * @param {HTMLElement} el
 * @param {string} text
 * @param {boolean} [isError=false]
 */
function setStatus(el, text, isError = false) {
  el.textContent = text;
  el.style.display = 'block';
  el.classList.toggle('upload-status--error', isError);
}

function clearStatus(el) {
  el.style.display = 'none';
  el.textContent = '';
  el.classList.remove('upload-status--error');
}
async function loadNpy(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error('Network response was not ok');
  const buffer = await response.arrayBuffer();
  const dataView = new DataView(buffer);
  
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 6));
  if (magic !== '\x93NUMPY') throw new Error('Not a valid .npy file');
  
  const major = dataView.getUint8(6);
  const headerLen = major === 1 ? dataView.getUint16(8, true) : dataView.getUint32(8, true);
  const headerStr = new TextDecoder().decode(new Uint8Array(buffer, major === 1 ? 10 : 12, headerLen));
  
  const dictMatch = headerStr.match(/'shape': \(([^)]+)\)/);
  if (!dictMatch) throw new Error('Could not parse shape from .npy header');
  const shape = dictMatch[1].split(',').map(x => parseInt(x.trim())).filter(x => !isNaN(x));
  
  const offset = (major === 1 ? 10 : 12) + headerLen;
  const float32Data = new Float32Array(buffer, offset);
  
  const rows = shape[0];
  const cols = shape.length > 1 ? shape[1] : 1;
  const matrix = [];
  for (let i = 0; i < rows; i++) {
    matrix.push(float32Data.subarray(i * cols, (i + 1) * cols));
  }
  return matrix;
}

function cosineSimilarity(A, B) {
  let dotProduct = 0, normA = 0, normB = 0;
  const len = Math.min(A.length, B.length);
  for (let i = 0; i < len; i++) {
    dotProduct += A[i] * B[i];
    normA += A[i] * A[i];
    normB += B[i] * B[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

let _cachedArtist = null;
let cachedEmbeddings = null;
let _currentSourceMode = 'yt'; // 'yt' | 'file'
let _currentlyEditingId = null; // ID of the song being edited inline

async function fetchExistingEmbeddings(artist) {
  if (cachedEmbeddings && _cachedArtist === artist) return cachedEmbeddings;
  
  _cachedArtist = artist;
  cachedEmbeddings = {};
  
  const fetchPromises = ALL_STEMS.map(async (stem) => {
    try {
      const vectors = await loadNpy(`./data/embeddings/${encodeURIComponent(artist)}/${stem}.npy`);
      return { stem, vectors };
    } catch (e) {
      console.warn(`Could not load embeddings for ${stem}:`, e);
      return { stem, vectors: [] };
    }
  });

  const results = await Promise.all(fetchPromises);
  
  results.forEach(({ stem, vectors }) => {
    cachedEmbeddings[stem] = vectors;
    if (DATA && DATA.songs && vectors.length > 0) {
      DATA.similarities[stem] = vectors.map((v1) => 
        vectors.map((v2) => cosineSimilarity(v1, v2))
      );
      if (!DATA.tsne[stem] || DATA.tsne[stem].length === 0) {
          DATA.tsne[stem] = new Array(vectors.length).fill([0, 0]);
      }
      if (!DATA.stem_energy[stem] || DATA.stem_energy[stem].length === 0) {
          DATA.stem_energy[stem] = new Array(vectors.length).fill(1.0);
      }
    }
  });

  return cachedEmbeddings;
}

async function getGeminiEmbedding(apiKey, base64Audio, mimeType="audio/wav") {
  const modelName = "models/gemini-embedding-2-preview";
  const url = `https://generativelanguage.googleapis.com/v1beta/${modelName}:embedContent?key=${apiKey}`;
  const res = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      model: modelName,
      content: {
        parts: [{ inlineData: { mimeType, data: base64Audio } }]
      }
    })
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Gemini API Error: ${err}`);
  }
  const data = await res.json();
  return data.embedding.values;
}

/**
 * Converts a "m:ss" or "mm:ss" or plain seconds string to seconds.
 * @param {string} str
 * @returns {number}
 */
function parseTimestamp(str) {
  const s = str.trim();
  if (s.includes(':')) {
    const [m, sec] = s.split(':').map(Number);
    return m * 60 + sec;
  }
  return Number(s) || 0;
}

/**
 * Slices an AudioBuffer between startSec and endSec and returns a WAV Blob.
 * Runs entirely in the browser — no server needed.
 * @param {Blob} audioBlob
 * @param {number} startSec
 * @param {number} endSec
 * @returns {Promise<Blob>}
 */
async function cutAudioClientSide(audioBlob, startSec, endSec) {
  const arrayBuffer = await audioBlob.arrayBuffer();
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const decoded = await ctx.decodeAudioData(arrayBuffer);
  await ctx.close();

  const sr = decoded.sampleRate;
  const channels = decoded.numberOfChannels;
  const startSample = Math.round(startSec * sr);
  const endSample   = Math.min(Math.round(endSec * sr), decoded.length);
  const frameCount  = endSample - startSample;

  if (frameCount <= 0) throw new Error('End time must be after start time.');

  const offCtx = new OfflineAudioContext(channels, frameCount, sr);
  const sliced = offCtx.createBuffer(channels, frameCount, sr);
  for (let c = 0; c < channels; c++) {
    sliced.copyToChannel(decoded.getChannelData(c).slice(startSample, endSample), c);
  }

  // Encode AudioBuffer → WAV Blob
  const interleaved = channels === 1
    ? sliced.getChannelData(0)
    : (() => {
        const out = new Float32Array(frameCount * channels);
        for (let i = 0; i < frameCount; i++)
          for (let c = 0; c < channels; c++)
            out[i * channels + c] = sliced.getChannelData(c)[i];
        return out;
      })();

  const byteCount = 44 + interleaved.length * 2;
  const buffer    = new ArrayBuffer(byteCount);
  const view      = new DataView(buffer);
  const writeStr  = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };

  writeStr(0, 'RIFF');
  view.setUint32(4,  byteCount - 8,  true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16,             true);
  view.setUint16(20, 1,              true); // PCM
  view.setUint16(22, channels,       true);
  view.setUint32(24, sr,             true);
  view.setUint32(28, sr * channels * 2, true);
  view.setUint16(32, channels * 2,   true);
  view.setUint16(34, 16,             true); // bit depth
  writeStr(36, 'data');
  view.setUint32(40, interleaved.length * 2, true);
  let off = 44;
  for (let i = 0; i < interleaved.length; i++, off += 2) {
    view.setInt16(off, Math.max(-1, Math.min(1, interleaved[i])) * 0x7FFF, true);
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

/** Toggle between YouTube and file-upload source modes. */
window.switchSourceTab = function(mode) {
  const sourceYt = document.getElementById('source-yt');
  const sourceFile = document.getElementById('source-file');
  
  if (mode === 'yt') {
    sourceYt.style.display = 'block';
    sourceFile.style.display = 'none';
  } else {
    sourceYt.style.display = 'none';
    sourceFile.style.display = 'block';
    // Clear YT URL when switching to File mode to avoid accidental inclusion
    document.getElementById('yt-url').value = '';
  }

  document.getElementById('tab-yt').classList.toggle('active',   mode === 'yt');
  document.getElementById('tab-file').classList.toggle('active', mode === 'file');
  document.getElementById('btn-download-icon').textContent  = mode === 'yt' ? '⬇' : '📂';
  document.getElementById('btn-download-label').textContent = mode === 'yt' ? 'Download Audio' : 'Use This File';
  document.getElementById('step1-title').textContent = mode === 'yt' ? 'Download from YouTube' : 'Upload Audio File';
  document.getElementById('step1-desc').textContent  = mode === 'yt'
    ? 'Paste a YouTube URL to download as MP3.'
    : 'Select or drop any audio file to use as the source.';
  _currentSourceMode = mode;
};

function setupUploadTab() {
  const btnDownload = document.getElementById('btn-download');
  if (!btnDownload) return;

  const btnCut      = document.getElementById('btn-cut');
  const btnSeparate = document.getElementById('btn-separate');
  const btnEmbed    = document.getElementById('btn-embed');

  const statusDownload  = document.getElementById('status-download');
  const statusCut       = document.getElementById('status-cut');
  const statusSeparate  = document.getElementById('status-separate');
  const statusEmbed     = document.getElementById('status-embed');

  let downloadedAudioBlob = null;
  let cutAudioBlob        = null;
  let stemsData           = null; // High Quality for embeddings
  let stemsDataLow        = null; // Low Quality for storage
  let currentSongInfo     = {};
  let currentTrimTimes    = { start: '0:00', end: '1:00' };
  let newSongEmbeddings   = {}; // Cache for resumable progress

  // ── File drop zone ───────────────────────────────────────────────────────────
  const fileInput  = document.getElementById('file-input');
  const dropZone   = document.getElementById('drop-zone');
  const fileLabel  = document.getElementById('drop-zone-filename');

  function handleAudioFile(file) {
    if (!file || !file.type.startsWith('audio/')) {
      setStatus(statusDownload, 'Please select a valid audio file.', true);
      return;
    }
    downloadedAudioBlob = file;
    fileLabel.textContent = `✓ ${file.name}`;
    const audioEl = document.getElementById('audio-preview-downloaded');
    audioEl.src = URL.createObjectURL(file);
    audioEl.style.display = 'block';
    setStatus(statusDownload, `File loaded: ${file.name} ✓`);
    newSongEmbeddings = {}; // Reset progress for new file
    showStep('step-2');
  }

  fileInput.addEventListener('change', () => fileInput.files[0] && handleAudioFile(fileInput.files[0]));
  dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) handleAudioFile(e.dataTransfer.files[0]);
  });

  // ── Sample Audio Logic ──
  const SAMPLES = {
    gehra_hua: {
      type: 'yt',
      url: 'https://www.youtube.com/watch?v=GX9x62kFsVU',
      name: 'Gehra Hua | Dhurandhar',
      singer: 'Arijit Singh',
      start: '0:00', end: '0:30'
    },
    phir_se: {
      type: 'yt',
      url: 'https://www.youtube.com/watch?v=Ans8Y59cvds',
      name: 'PHIR SE | Dhurandhar The Revenge',
      singer: 'Arijit Singh',
      start: '0:00', end: '0:30'
    },
    sitaare: {
      type: 'yt',
      url: 'https://www.youtube.com/watch?v=nDjloeIB3Pc',
      name: 'Sitaare | Ikkis',
      singer: 'Arijit Singh',
      start: '0:00', end: '0:30'
    },
    sajni: {
      type: 'file',
      path: './data/samples/sajni.mp3',
      name: 'Sajni | Laapataa Ladies',
      singer: 'Arijit Singh',
      start: '0:00', end: '0:30'
    },
    laal_ishq: {
      type: 'file',
      path: './data/samples/laal_ishq.mp3',
      name: 'Laal Ishq | Ram-leela',
      singer: 'Arijit Singh',
      start: '1:00', end: '1:30'
    },
    aayat: {
      type: 'file',
      path: './data/samples/aayat.mp3',
      name: 'Aayat | Bajirao Mastani',
      singer: 'Arijit Singh',
      start: '0:45', end: '1:15'
    },
    fate_of_ophelia: {
      type: 'yt',
      url: 'https://www.youtube.com/watch?v=ko70cExuzZM',
      name: 'The Fate of Ophelia',
      singer: 'Taylor Swift',
      start: '0:00', end: '0:30'
    },
    opalite_yt: {
      type: 'yt',
      url: 'https://www.youtube.com/watch?v=1FVF-9KQiPo',
      name: 'Opalite',
      singer: 'Taylor Swift',
      start: '0:00', end: '0:30'
    },
    anti_hero: {
      type: 'file',
      path: './data/samples/anti-hero.mp3',
      name: 'Anti-Hero',
      singer: 'Taylor Swift',
      start: '0:00', end: '0:30'
    },
    karma: {
      type: 'file',
      path: './data/samples/ice_spice_-_karma.mp3',
      name: 'Karma (ft. Ice Spice)',
      singer: 'Taylor Swift',
      start: '0:00', end: '0:30'
    }
  };

  document.querySelectorAll('.sample-chip-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const id = btn.dataset.id;
      const type = btn.dataset.type;
      const s = SAMPLES[id];
      if (!s) return;

      try {
        if (type === 'yt') {
          document.getElementById('yt-url').value = s.url;
          document.getElementById('yt-song-name').value = s.name;
          const artistSelector = document.getElementById('yt-singer');
          if (artistSelector) artistSelector.value = s.singer;
          
          currentSongInfo.singer = s.singer;
          currentSongInfo.songName = s.name;
          
          setStatus(statusDownload, `Sample URL loaded: ${s.name} ✓`);
          // Note: Step 2 is NOT shown yet because they still need to click "Download" to fetch it
        } else {
          setStatus(statusDownload, `Loading sample file: ${s.name}...`);
          const response = await fetch(s.path);
          if (!response.ok) throw new Error(`Could not find ${s.path}`);
          const blob = await response.blob();
          const file = new File([blob], `${id}.mp3`, { type: 'audio/mpeg' });
          handleAudioFile(file);
          
          document.getElementById('yt-song-name').value = s.name;
          const artistSelector = document.getElementById('yt-singer');
          if (artistSelector) artistSelector.value = s.singer;

          currentSongInfo.singer = s.singer;
          currentSongInfo.songName = s.name;
          
          setStatus(statusDownload, `Sample file loaded: ${s.name} ✓`);
        }

        // Set suggested trim
        document.getElementById('cut-start').value = s.start;
        document.getElementById('cut-end').value = s.end;
        updateTrimDuration();

      } catch (e) {
        setStatus(statusDownload, `Error loading sample: ${e.message}`, true);
      }
    });
  });

  // ── Trim duration display ────────────────────────────────────────────────────
  function updateTrimDuration() {
    const start = parseTimestamp(document.getElementById('cut-start').value);
    const end   = parseTimestamp(document.getElementById('cut-end').value);
    const dur   = Math.max(0, end - start);
    const durationEl = document.getElementById('trim-duration');
    if (durationEl) durationEl.textContent = `Duration: ${dur}s`;
  }
  document.getElementById('cut-start').addEventListener('input', updateTrimDuration);
  document.getElementById('cut-end').addEventListener('input', updateTrimDuration);

  // ── Step 1: Source ───────────────────────────────────────────────────────────
  btnDownload.addEventListener('click', async () => {
    const songNameVal = document.getElementById('yt-song-name').value.trim();
    const newSinger   = document.getElementById('yt-new-singer').value.trim();
    const singer      = newSinger || document.getElementById('yt-singer').value;

    currentSongInfo = { songName: songNameVal || 'Unknown Song', singer };

    // File mode: already handled by handleAudioFile — just confirm metadata
    if (_currentSourceMode === 'file') {
      if (!downloadedAudioBlob) { setStatus(statusDownload, 'Please select an audio file first.', true); return; }
      setStatus(statusDownload, `Using: ${downloadedAudioBlob.name || 'uploaded file'} ✓`);
      newSongEmbeddings = {}; // Reset progress
      showStep('step-2');
      return;
    }

    // YouTube mode
    const url = document.getElementById('yt-url').value.trim();
    if (!url) { setStatus(statusDownload, 'Please provide a YouTube URL.', true); return; }

    setStatus(statusDownload, 'Downloading from YouTube…');
    btnDownload.disabled = true;

    try {
      const SAMPLE_URLS = {
        "https://www.youtube.com/watch?v=GX9x62kFsVU": { file: "gehra_hua.mp3", title: "Gehra Hua | Dhurandhar" },
        "https://www.youtube.com/watch?v=Ans8Y59cvds": { file: "phir_se.mp3", title: "PHIR SE | Dhurandhar The Revenge" },
        "https://www.youtube.com/watch?v=nDjloeIB3Pc": { file: "sitaare.mp3", title: "Sitaare | Ikkis" },
        "https://www.youtube.com/watch?v=ko70cExuzZM": { file: "the_fate_of_ophelia.mp3", title: "The Fate of Ophelia" },
        "https://www.youtube.com/watch?v=1FVF-9KQiPo": { file: "opalite.mp3", title: "Opalite" },
        "https://www.youtube.com/watch?v=b1kbLwvqugk": { file: "anti-hero.mp3", title: "Anti-Hero" },
        "https://www.youtube.com/watch?v=XzOvgu3GPwY": { file: "ice_spice_-_karma.mp3", title: "Karma (ft. Ice Spice)" }
      };

      let res, title;
      if (SAMPLE_URLS[url]) {
          const info = SAMPLE_URLS[url];
          res = await fetch(`./data/samples/${info.file}`);
          if (!res.ok) throw new Error("Sample file not found in local data folder.");
          title = info.title;
      } else {
          const formData = new FormData();
          formData.append('url', url);
          res = await fetch(`${BACKEND_URL}/api/download`, { method: 'POST', body: formData });
          if (!res.ok) {
              const errText = await res.text();
              if (errText.includes('<!DOCTYPE html>')) throw new Error(`Server Error (${res.status}). Contact the developer.`);
              throw new Error(errText.substring(0, 100) + 'Contact the developer.');
          }
          title = res.headers.get('X-Audio-Title');
      }

      if (title && !songNameVal) currentSongInfo.songName = title;
      downloadedAudioBlob = await res.blob();
      const audioEl = document.getElementById('audio-preview-downloaded');
      audioEl.src = URL.createObjectURL(downloadedAudioBlob);
      audioEl.style.display = 'block';

      setStatus(statusDownload, 'Downloaded successfully! ✓');
      showStep('step-2');
    } catch (e) {
      setStatus(statusDownload, `Error: ${e.message}`, true);
    } finally {
      btnDownload.disabled = false;
    }
  });

  // ── Step 2: Trim (client-side, no server) ────────────────────────────────────
  btnCut.addEventListener('click', async () => {
    if (!downloadedAudioBlob) return;
    const start = document.getElementById('cut-start').value.trim() || '0:00';
    const end   = document.getElementById('cut-end').value.trim() || '1:00';
    const startSec = parseTimestamp(start);
    const endSec   = parseTimestamp(end);

    if (endSec - startSec > 60) { setStatus(statusCut, 'Max duration is 60 seconds.', true); return; }
    if (endSec <= startSec)     { setStatus(statusCut, 'End time must be after start time.', true); return; }

    currentTrimTimes = { start, end };
    setStatus(statusCut, 'Trimming audio in browser…');
    btnCut.disabled = true;

    try {
      cutAudioBlob = await cutAudioClientSide(downloadedAudioBlob, startSec, endSec);

      const audioEl = document.getElementById('audio-preview-cut');
      audioEl.src = URL.createObjectURL(cutAudioBlob);
      audioEl.style.display = 'block';

      setStatus(statusCut, `Trimmed to ${endSec - startSec}s ✓`);
      showStep('step-3');
    } catch (e) {
      setStatus(statusCut, `Error: ${e.message}`, true);
    } finally {
      btnCut.disabled = false;
    }
  });

  // ── Step 3: Separate ─────────────────────────────────────────────────────────
  btnSeparate.addEventListener('click', async () => {
    if (!cutAudioBlob) return;
    setStatus(statusSeparate, 'Separating stems with Demucs — this may take a few minutes…');
    btnSeparate.disabled = true;
    try {
      const formData = new FormData();
      formData.append('file', cutAudioBlob, 'cut.wav');
      newSongEmbeddings = {}; // Reset progress for new separation
      const res = await fetch(`${BACKEND_URL}/api/separate`, { method: 'POST', body: formData });
      if (!res.ok) {
          const errText = await res.text();
          if (errText.includes('<!DOCTYPE html>')) throw new Error(`Server Error (${res.status})`);
          throw new Error(errText.substring(0, 100));
      }
      const data = await res.json();
      stemsData = data.stems;       // HQ
      stemsDataLow = data.stems_low; // LQ
      setStatus(statusSeparate, 'Stems separated successfully! ✓');
      showStep('step-4');
    } catch (e) {
      setStatus(statusSeparate, `Error: ${e.message}`, true);
    } finally {
      btnSeparate.disabled = false;
    }
  });

  // ── Step 4: Embed ─────────────────────────────────────────────────────────────
  btnEmbed.addEventListener('click', async () => {
    const apiKey = document.getElementById('gemini-key').value.trim();
    if (!apiKey)    { setStatus(statusEmbed, 'Please provide a Gemini API Key.', true); return; }
    if (!stemsData) { setStatus(statusEmbed, 'No stems available. Please complete step 3 first.', true); return; }

    setStatus(statusEmbed, 'Generating embeddings via Gemini…');
    btnEmbed.disabled = true;

    try {
      await fetchExistingEmbeddings(currentSongInfo.singer);
      
      for (const stem of ALL_STEMS) {
        if (!stemsData[stem]) continue;
        if (newSongEmbeddings[stem]) {
            console.log(`Skipping already embedded stem: ${stem}`);
            continue;
        }

        setStatus(statusEmbed, `Generating embedding for stem: ${stem} (${ALL_STEMS.indexOf(stem) + 1}/${ALL_STEMS.length})…`);
        // Use HQ MP3 stems for best quality embeddings
        newSongEmbeddings[stem] = await getGeminiEmbedding(apiKey, stemsData[stem], 'audio/mpeg');
      }

      setStatus(statusEmbed, 'Calculating similarities and updating library…');
      const newSongIdx   = DATA.songs.length;
      const songFilename = 'ADDED_' + currentSongInfo.songName.replace(/\s+/g, '_');

      const songObj = {
        index: newSongIdx,
        artist_group: currentSongInfo.singer,
        filename: songFilename,
        song_name: currentSongInfo.songName,
        film: 'User Added',
        year: new Date().getFullYear().toString(),
        primary_mood: '', secondary_moods: '',
        singer: currentSongInfo.singer, composer: '',
        youtube_url: _currentSourceMode === 'yt' ? document.getElementById('yt-url').value : '',
        youtube_views: 0,
        is_local: true,
        trim: currentTrimTimes,
      };
      DATA.songs.push(songObj);

      ALL_STEMS.forEach(stem => {
        if (!DATA.stem_energy[stem])   DATA.stem_energy[stem]   = [];
        if (!DATA.similarities[stem])  DATA.similarities[stem]  = [];
        DATA.stem_energy[stem].push(1.0);

        const newEmb = newSongEmbeddings[stem] || new Array(3072).fill(0);
        if (cachedEmbeddings[stem]) cachedEmbeddings[stem].push(new Float32Array(newEmb));

        const newSimRow = [];
        for (let i = 0; i < newSongIdx; i++) {
          const sim = (cachedEmbeddings[stem]?.[i] && newEmb)
            ? cosineSimilarity(cachedEmbeddings[stem][i], newEmb) : 0;
          newSimRow.push(sim);
          if (DATA.similarities[stem][i]) DATA.similarities[stem][i].push(sim);
        }
        newSimRow.push(1.0);
        DATA.similarities[stem].push(newSimRow);
      });

      ALL_STEMS.forEach(stem => { 
        if (DATA.tsne[stem]) {
          let bestIdx = 0;
          let maxSim = -Infinity;
          // Find the most similar existing song to place it nearby in the graph
          for (let i = 0; i < newSongIdx; i++) {
            const s = DATA.similarities[stem][newSongIdx][i];
            if (s > maxSim) { maxSim = s; bestIdx = i; }
          }
          const [nx, ny] = DATA.tsne[stem][bestIdx] || [0, 0];
          // Add a tiny random jitter to avoid perfect overlap
          DATA.tsne[stem].push([nx + (Math.random() - 0.5) * 3, ny + (Math.random() - 0.5) * 3]);
        }
      });

      // Store LQ stems in library to save space
      storeLocalSong(songObj, stemsDataLow, newSongEmbeddings,
        Object.fromEntries(ALL_STEMS.map(s => [s, [...DATA.similarities[s][newSongIdx]]])));

      setStatus(statusEmbed, '✓ Song added to library!');
      setTimeout(() => clearStatus(statusEmbed), 4000);

      renderSimilarityList();
      buildCompareSelects();
      renderAddedSongs();

      // Reset pipeline
      document.getElementById('yt-url').value = '';
      document.getElementById('yt-song-name').value = '';
      document.getElementById('yt-new-singer').value = '';
      document.getElementById('cut-start').value = '0:00';
      document.getElementById('cut-end').value = '1:00';
      
      const prev1 = document.getElementById('audio-preview-downloaded');
      prev1.src = ''; prev1.style.display = 'none';
      const prev2 = document.getElementById('audio-preview-cut');
      prev2.src = ''; prev2.style.display = 'none';
      
      clearStatus(statusDownload);
      clearStatus(statusCut);
      clearStatus(statusSeparate);
      
      showStep('step-1'); // Go back to start
      showStep('step-2', false);
      showStep('step-3', false);
      showStep('step-4', false);
      
      downloadedAudioBlob = null;
      cutAudioBlob = null;
      stemsData = null;
      stemsDataLow = null;
      newSongEmbeddings = {};
      currentTrimTimes = { start: '0:00', end: '1:00' };

    } catch (e) {
      console.error(e);
      setStatus(statusEmbed, `Error: ${e.message}`, true);
    } finally {
      btnEmbed.disabled = false;
    }
  });

  renderAddedSongs();
}

const dbName = "SonicLibrary";

const storeName = "added_songs";

function initDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(dbName, 1);
        request.onupgradeneeded = event => {
            const db = event.target.result;
            if (!db.objectStoreNames.contains(storeName)) {
                db.createObjectStore(storeName, { keyPath: "id", autoIncrement: true });
            }
        };
        request.onsuccess = event => resolve(event.target.result);
        request.onerror = event => reject(event.target.error);
    });
}

async function getAllAddedSongs() {
    try {
        const db = await initDB();
        return await new Promise((resolve, reject) => {
            const tx = db.transaction(storeName, 'readonly');
            const req = tx.objectStore(storeName).getAll();
            req.onsuccess = () => resolve(req.result || []);
            req.onerror = () => reject(req.error);
        });
    } catch {
        return [];
    }
}

window.deleteLocalSong = async function(id) {
    if (!confirm('Are you sure you want to delete this song from your local library?')) return;
    try {
        const db = await initDB();
        const tx = db.transaction(storeName, "readwrite");
        tx.objectStore(storeName).delete(id);
        tx.oncomplete = () => {
            _addedSongCache.clear();
            renderAddedSongs();
            alert('Song deleted. Please reload the page to completely clear it from the session data.');
        };
    } catch(e) {
        console.error("Failed to delete song:", e);
    }
}

window.playLocalSong = async function(id) {
    const songs = await getAllAddedSongs();
    const song = songs.find(s => s.id === id);
    if (song) {
        playAudio(song.songObj.filename, 'original', song.songObj.song_name);
    }
}

window.toggleEditInline = function(id) {
    _currentlyEditingId = (_currentlyEditingId === id) ? null : id;
    renderAddedSongs();
}

window.saveEditInline = async function(id) {
    const songs = await getAllAddedSongs();
    const song = songs.find(s => s.id === id);
    if (!song) return;

    const newName = document.getElementById(`edit-name-${id}`).value.trim();
    const newSinger = document.getElementById(`edit-singer-${id}`).value.trim();

    if (!newName || !newSinger) return;

    try {
        const db = await initDB();
        const tx = db.transaction(storeName, "readwrite");
        const store = tx.objectStore(storeName);
        
        song.songObj.song_name = newName;
        song.songObj.singer = newSinger;
        song.songObj.artist_group = newSinger;
        
        const existingInData = DATA.songs.find(s => s.filename === song.songObj.filename);
        if (existingInData) {
            existingInData.song_name = newName;
            existingInData.singer = newSinger;
            existingInData.artist_group = newSinger;
        }

        store.put(song);
        tx.oncomplete = () => {
            _addedSongCache.clear();
            _currentlyEditingId = null;
            renderAddedSongs();
            renderSimilarityList();
            buildCompareSelects();
        };
    } catch(e) {
        console.error("Failed to edit song:", e);
    }
}

async function getLibraryStorageUsage() {
    const songs = await getAllAddedSongs();
    let totalBytes = 0;
    songs.forEach(item => {
        // Approximate size of stems (base64)
        if (item.stems) {
            Object.values(item.stems).forEach(val => {
                if (val) totalBytes += val.length * 0.75; // base64 to bytes
            });
        }
    });
    
    if (totalBytes < 1024 * 1024) return (totalBytes / 1024).toFixed(1) + ' KB';
    return (totalBytes / (1024 * 1024)).toFixed(1) + ' MB';
}

async function storeLocalSong(songObj, stems, embeddings, similarities) {
    try {
        const db = await initDB();
        const tx = db.transaction(storeName, "readwrite");
        tx.objectStore(storeName).add({ songObj, stems, embeddings, similarities });
        _addedSongCache.clear(); // invalidate cache so getAddedSong refetches
    } catch(e) {
        console.error("Failed to store song locally:", e);
    }
}

async function renderAddedSongs() {
    const container = document.getElementById('added-songs-list');
    if (!container) return;
    container.innerHTML = '';

    try {
        const songs = await getAllAddedSongs();

        // Update sidebar count and storage badge
        const countEl = document.getElementById('added-songs-count');
        if (countEl) {
            const usage = await getLibraryStorageUsage();
            countEl.innerHTML = `${songs.length} tracks <span style="opacity:0.6;margin-left:0.4rem;font-weight:400">(${usage})</span>`;
        }

        if (songs.length === 0) {
            container.innerHTML = '<p class="added-songs-empty">No songs added yet.</p>';
            return;
        }

        songs.forEach(item => {
            const isEditing = _currentlyEditingId === item.id;
            const div = document.createElement('div');
            div.className = 'added-song-card' + (isEditing ? ' editing' : '');
            
            if (isEditing) {
                div.innerHTML = `
                    <div class="added-song-edit-form">
                        <input type="text" id="edit-name-${item.id}" class="upload-input small" value="${item.songObj.song_name}" placeholder="Song Name">
                        <input type="text" id="edit-singer-${item.id}" class="upload-input small" value="${item.songObj.singer}" placeholder="Singer">
                        <div class="added-song-edit-actions">
                            <button class="action-link save" onclick="saveEditInline(${item.id})">Save</button>
                            <button class="action-link cancel" onclick="toggleEditInline(null)">Cancel</button>
                        </div>
                    </div>
                `;
            } else {
                div.innerHTML = `
                    <div class="added-song-info">
                        <div class="added-song-title">${item.songObj.song_name}</div>
                        <div class="added-song-meta">
                            ${item.songObj.singer}
                            ${item.songObj.trim ? `<span class="added-song-trim">(${item.songObj.trim.start} – ${item.songObj.trim.end})</span>` : ''}
                        </div>
                    </div>
                    <div class="added-song-actions">
                        <button class="icon-btn play-btn-small" onclick="playLocalSong(${item.id})" title="Play Song">▶</button>
                        <button class="icon-btn edit-btn-small" onclick="toggleEditInline(${item.id})" title="Edit Info">✎</button>
                        <button class="icon-btn delete-btn-small" onclick="deleteLocalSong(${item.id})" title="Delete Song">✕</button>
                    </div>
                `;
            }
            container.appendChild(div);
        });
    } catch (e) {
        console.error('Failed to render local songs:', e);
    }
}

/* ── URL State Sync ── */
window.addEventListener('popstate', () => { parseURLState(); location.reload(); });

function parseURLState() {
  const params = new URLSearchParams(window.location.search);
  clusterThresholdFromURL = params.has('threshold');
  if (params.has('tab')) {
    const tabName = params.get('tab');
    const tabBtn = document.querySelector(`.tab-btn[data-tab="${tabName}"]`);
    if (tabBtn) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
      tabBtn.classList.add('active');
      document.getElementById('panel-' + tabName).classList.add('active');
    }
  }
  if (params.has('stem')) {
    const s = params.get('stem');
    if (ALL_STEMS.includes(s)) currentStem = s;
  }
  if (params.has('song')) {
    const n = parseInt(params.get('song'));
    if (!isNaN(n) && DATA && n >= 0 && n < DATA.songs.length) selectedSongIdx = n;
  }
  if (params.has('threshold')) {
    clusterThresholdFromURL = true;
    const t = parseFloat(params.get('threshold'));
    if (!isNaN(t) && t >= 0 && t <= 100) {
      const slider = document.getElementById('sim-threshold');
      if (slider) {
        slider.value = t.toFixed(1);
        const valEl = document.getElementById('sim-threshold-val');
        if (valEl) valEl.textContent = t.toFixed(1) + '%';
      }
    }
  }
}

function updateURLState() {
  const params = new URLSearchParams(window.location.search);
  const activeTab = document.querySelector('.tab-btn.active');
  if (activeTab) params.set('tab', activeTab.dataset.tab);
  params.set('stem', currentStem);
  if (selectedSongIdx !== null) params.set('song', selectedSongIdx);
  const slider = document.getElementById('sim-threshold');
  if (slider) params.set('threshold', slider.value);
  window.history.replaceState(null, '', window.location.pathname + '?' + params.toString());
}

/* ── Tabs ── */
function setupTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn =>
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('panel-' + btn.dataset.tab).classList.add('active');
      
      updateURLState();
      
      if (btn.dataset.tab === 'compare') renderCompare();
      if (btn.dataset.tab === 'heatmap') renderHeatmap();
      if (btn.dataset.tab === 'cluster') renderCluster();
    })
  );
}

/* ── Stem Bars ── */
function buildAllStemBars() {
  ['sim-stems','heatmap-stems','cluster-stems'].forEach(id => {
    const c = document.getElementById(id);
    if (!c) return;
    ALL_STEMS.forEach(stem => {
      const chip = document.createElement('div');
      chip.className = 'stem-chip' + (stem === currentStem ? ' active' : '');
      chip.dataset.stem = stem;
      chip.textContent = stem;
      chip.addEventListener('click', () => switchStem(stem));
      c.appendChild(chip);
    });
  });
}

function switchStem(stem) {
  if (customClusterMode) {
    customClusterMode = false;
    customClusterSimilarityMatrix = null;
    closeCustomClusterPanel();
  }
  currentStem = stem;
  document.querySelectorAll('.stem-chip').forEach(c => {
    c.classList.toggle('active', c.dataset.stem === stem);
  });
  updateURLState();
  if (document.getElementById('panel-similarity').classList.contains('active')) renderSimilarityResults(selectedSongIdx || 0);
  if (document.getElementById('panel-compare').classList.contains('active')) renderCompare();
  if (document.getElementById('panel-heatmap').classList.contains('active')) renderHeatmap();
  if (document.getElementById('panel-cluster').classList.contains('active')) renderCluster();
}


/* ── Helpers ── */
function slugify(text) {
  if (!text) return '';
  return text.toLowerCase().trim()
    .replace(/[^\w\s-]/g, '')
    .replace(/[\s_-]+/g, '-');
}
function songName(s) { return s.song_name; }
function esc(s) { return s.replace(/'/g, "\\'"); }
function stemEnergy(songIdx, stem) {
  return DATA.stem_energy?.[stem]?.[songIdx] ?? 1;
}
function simColor(v) {
  if (v >= 0.85) return '#22c55e'; if (v >= 0.7) return '#4ade80';
  if (v >= 0.5) return '#eab308'; if (v >= 0.3) return '#f97316';
  return '#ef4444';
}
function getSimilarity(stem, i, j) {
  const row = DATA?.similarities?.[stem]?.[i];
  if (row && Number.isFinite(row[j])) return row[j];
  const reverseRow = DATA?.similarities?.[stem]?.[j];
  if (reverseRow && Number.isFinite(reverseRow[i])) return reverseRow[i];
  return 0;
}
const YOUTUBE_NODE_SIZES = [3.0, 4.1, 5.4, 6.8];
function formatYoutubeViews(views) {
  if (!Number.isFinite(views)) return '—';
  if (views >= 1e9) return `${(views / 1e9).toFixed(views >= 1e10 ? 0 : 1)}B`;
  if (views >= 1e6) return `${(views / 1e6).toFixed(views >= 1e7 ? 0 : 1)}M`;
  if (views >= 1e3) return `${(views / 1e3).toFixed(views >= 1e4 ? 0 : 1)}K`;
  return `${Math.round(views)}`;
}
function getYoutubeViewMeta() {
  if (!DATA?.songs?.length) return null;

  const views = DATA.songs
    .map(song => Number(song.youtube_views))
    .filter(value => Number.isFinite(value) && value > 0)
    .sort((a, b) => a - b);

  if (!views.length) return null;

  const q = (pct) => views[Math.min(views.length - 1, Math.max(0, Math.floor((views.length - 1) * pct)))];
  const min = views[0];
  const q1 = q(0.25);
  const q2 = q(0.5);
  const q3 = q(0.75);
  const max = views[views.length - 1];

  return {
    min,
    max,
    thresholds: [q1, q2, q3],
    buckets: [
      { label: 'Low', min, max: q1, size: YOUTUBE_NODE_SIZES[0] },
      { label: 'Mid', min: q1, max: q2, size: YOUTUBE_NODE_SIZES[1] },
      { label: 'High', min: q2, max: q3, size: YOUTUBE_NODE_SIZES[2] },
      { label: 'Top', min: q3, max, size: YOUTUBE_NODE_SIZES[3] },
    ],
  };
}
function getYoutubeViewBucket(songIdx) {
  const song = DATA?.songs?.[songIdx];
  const views = Number(song?.youtube_views);
  if (!Number.isFinite(views) || views <= 0) return null;

  const meta = getYoutubeViewMeta();
  if (!meta) return null;

  const [q1, q2, q3] = meta.thresholds;
  if (views <= q1) return 0;
  if (views <= q2) return 1;
  if (views <= q3) return 2;
  return 3;
}
function buildCustomClusterSimilarityMatrix(weights = customClusterWeights) {
  const n = DATA?.songs?.length || 0;
  if (!n) return null;

  const matrix = Array.from({ length: n }, () => Array(n).fill(0));
  const totalWeight = PLAYABLE.reduce((sum, stem) => sum + Math.max(0, Number(weights[stem]) || 0), 0);
  if (totalWeight <= 0) return matrix;

  PLAYABLE.forEach(stem => {
    const weight = Math.max(0, Number(weights[stem]) || 0);
    const sims = DATA?.similarities?.[stem];
    if (!weight || !sims) return;

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        matrix[i][j] += sims[i][j] * weight;
      }
    }
  });

  return matrix.map(row => row.map(value => value / totalWeight));
}
function getClusterSimilarityMatrix() {
  if (customClusterMode) {
    if (!customClusterSimilarityMatrix) customClusterSimilarityMatrix = buildCustomClusterSimilarityMatrix();
    return customClusterSimilarityMatrix;
  }
  return DATA?.similarities?.[currentStem] ?? null;
}
function getClusterThresholdMetaFromMatrix(sims) {
  if (!sims || !sims.length) return { min: 0, max: 100, thresholds: [0, 100] };

  const edges = [];
  for (let i = 0; i < sims.length; i++) {
    for (let j = i + 1; j < sims[i].length; j++) {
      edges.push({ i, j, value: Number((sims[i][j] * 100).toFixed(1)) });
    }
  }

  if (!edges.length) return { min: 0, max: 100, thresholds: [0, 100] };

  edges.sort((a, b) => b.value - a.value);

  const parent = Array.from({ length: sims.length }, (_, i) => i);
  const rank = Array.from({ length: sims.length }, () => 0);
  const find = (x) => {
    if (parent[x] !== x) parent[x] = find(parent[x]);
    return parent[x];
  };
  const union = (a, b) => {
    let rootA = find(a);
    let rootB = find(b);
    if (rootA === rootB) return false;
    if (rank[rootA] < rank[rootB]) [rootA, rootB] = [rootB, rootA];
    parent[rootB] = rootA;
    if (rank[rootA] === rank[rootB]) rank[rootA]++;
    return true;
  };

  const thresholds = [];
  for (const edge of edges) {
    if (union(edge.i, edge.j)) {
      thresholds.push(edge.value);
    }
  }

  const dedupedThresholds = Array.from(new Set(thresholds)).sort((a, b) => a - b);
  const actualMin = dedupedThresholds[0];
  const max = dedupedThresholds[dedupedThresholds.length - 1];
  const min = Math.max(0, Number((actualMin - 0.1).toFixed(1)));
  return {
    min,
    max,
    thresholds: [min, ...dedupedThresholds]
  };
}
function getThresholdPercentile(thresholds, percentile) {
  if (!thresholds || !thresholds.length) return 0;
  const index = Math.min(thresholds.length - 1, Math.max(0, Math.floor((thresholds.length - 1) * percentile)));
  return thresholds[index];
}
function snapToNearestThreshold(value, thresholds) {
  if (!thresholds || !thresholds.length || !Number.isFinite(value)) return value;
  return thresholds.reduce((nearest, current) => (
    Math.abs(current - value) < Math.abs(nearest - value) ? current : nearest
  ), thresholds[0]);
}
function syncClusterSliderBounds() {
  const slider = document.getElementById('sim-threshold');
  const valEl = document.getElementById('sim-threshold-val');
  if (!slider || !valEl) return;

  const { min, max, thresholds } = getClusterThresholdMetaFromMatrix(getClusterSimilarityMatrix());
  slider.min = min.toFixed(1);
  slider.max = max.toFixed(1);
  slider.step = 'any';

  const defaultValue = getThresholdPercentile(thresholds, 0.80);
  const current = Math.min(max, Math.max(min, parseFloat(slider.value)));
  const snapped = clusterThresholdFromURL || clusterSliderDefaulted
    ? snapToNearestThreshold(Number.isFinite(current) ? current : defaultValue, thresholds)
    : defaultValue;
  slider.value = snapped.toFixed(1);
  valEl.textContent = snapped.toFixed(1) + '%';
  clusterSliderDefaulted = true;
}
function clusterNodeColor(d) {
  if (!d.active || d.linkCount === 0) return '#6b7280';
  if (d.linkCount === 1) return '#22c55e';
  if (d.linkCount === 2) return '#eab308';
  return '#ef4444';
}
function clusterNodeRadius(d) {
  const bucket = getYoutubeViewBucket(d.id);
  if (bucket !== null) return YOUTUBE_NODE_SIZES[bucket] * _clusterRadiusScale;

  const jitter = (d.id % 7) * 0.1;
  if (!d.active || d.linkCount === 0) return (2.3 + jitter) * _clusterRadiusScale;
  return (2.8 + Math.min(d.linkCount, 8) * 0.55 + jitter) * _clusterRadiusScale;
}
function clusterLinkColor(sim) {
  const simPercent = Number((sim * 100).toFixed(1));
  if (simPercent >= 99.0) return 'rgba(239,68,68,0.78)';
  if (simPercent >= 85.0) return 'rgba(249,115,22,0.6)';
  return 'rgba(255,255,255,0.12)';
}
function clusterNodeStatusLabel(d) {
  if (!d.active || d.linkCount === 0) return 'isolated';
  if (d.linkCount === 1) return 'leaf';
  if (d.linkCount === 2) return 'bridge';
  return 'hub';
}
function formatCustomWeight(value) {
  return Number(value).toFixed(1);
}
function buildCustomClusterModalHTML() {
  const activeWeights = customClusterWeights || {};
  const formulaPreview = PLAYABLE
    .map(stem => `${stem}_sim × ${formatCustomWeight(activeWeights[stem] ?? 0)}`)
    .join(' + ');
  const weightSum = PLAYABLE.reduce((sum, stem) => sum + Math.max(0, Number(activeWeights[stem]) || 0), 0);

  return `
    <h2>Custom Similarity</h2>
    <div class="modal-sub">Choose stem weightages from 0 to 1. The cluster graph will use the weighted average of those stem similarities.</div>

    <div class="modal-section">
      <h4>Weight Controls</h4>
      <div style="display:grid;gap:0.75rem">
        ${PLAYABLE.map(stem => `
          <div style="display:grid;grid-template-columns:72px 1fr 38px;align-items:center;gap:0.5rem">
            <span style="font-size:0.76rem;font-weight:700;color:${STEM_COLORS[stem]};text-transform:capitalize">${stem}</span>
            <input
              id="custom-weight-${stem}"
              type="range"
              min="0"
              max="1"
              step="0.1"
              value="${formatCustomWeight(activeWeights[stem] ?? 1)}"
              oninput="updateCustomStemWeight('${stem}', this.value)"
            >
            <span id="custom-weight-value-${stem}" style="font-family:'JetBrains Mono';font-size:0.72rem;color:var(--text-muted)">${formatCustomWeight(activeWeights[stem] ?? 1)}</span>
          </div>
        `).join('')}
      </div>
    </div>

    <div class="modal-section">
      <h4>Formula Preview</h4>
      <div id="custom-formula-preview" style="font-family:'JetBrains Mono';font-size:0.72rem;line-height:1.7;color:var(--text-secondary);word-break:break-word">(${formulaPreview}) / ${weightSum.toFixed(1)}</div>
    </div>

    <div class="modal-actions" style="margin-top:1rem">
      <button class="modal-action-btn primary" onclick="applyCustomClusterWeights()">Apply to Cluster</button>
      <button class="modal-action-btn" onclick="disableCustomClusterMode()">Use Stem Similarity</button>
    </div>
  `;
}
function buildCustomClusterPanelHTML() {
  const activeWeights = customClusterWeights || {};
  const formulaPreview = PLAYABLE
    .map(stem => `${stem}_sim × ${formatCustomWeight(activeWeights[stem] ?? 0)}`)
    .join(' + ');
  const weightSum = PLAYABLE.reduce((sum, stem) => sum + Math.max(0, Number(activeWeights[stem]) || 0), 0);

  return `
    <div style="display:flex;align-items:center;justify-content:space-between;gap:0.75rem;margin-bottom:0.55rem">
      <div>
        <div style="font-size:0.76rem;font-weight:700;color:#e8e8f0">Custom Similarity</div>
        <div style="font-size:0.64rem;color:var(--text-muted)">Set stem weights from 0 to 1</div>
      </div>
      <button class="stem-chip" onclick="closeCustomClusterPanel()">Close</button>
    </div>

    <div style="display:grid;gap:0.6rem">
      ${PLAYABLE.map(stem => `
        <div style="display:grid;grid-template-columns:72px 1fr 34px;align-items:center;gap:0.45rem">
          <span style="font-size:0.72rem;font-weight:700;color:${STEM_COLORS[stem]};text-transform:capitalize">${stem}</span>
          <input
            id="custom-weight-${stem}"
            type="range"
            min="0"
            max="1"
            step="0.1"
            value="${formatCustomWeight(activeWeights[stem] ?? 1)}"
            oninput="updateCustomStemWeight('${stem}', this.value)"
          >
          <span id="custom-weight-value-${stem}" style="font-family:'JetBrains Mono';font-size:0.66rem;color:var(--text-muted)">${formatCustomWeight(activeWeights[stem] ?? 1)}</span>
        </div>
      `).join('')}
    </div>

    <div style="margin-top:0.55rem;padding-top:0.5rem;border-top:1px solid rgba(255,255,255,0.06)">
      <div style="font-size:0.64rem;font-weight:700;color:var(--text-secondary);margin-bottom:0.2rem">Formula Preview</div>
      <div id="custom-formula-preview" style="font-family:'JetBrains Mono';font-size:0.66rem;line-height:1.45;color:var(--text-secondary);word-break:break-word">(${formulaPreview}) / ${weightSum.toFixed(1)}</div>
    </div>

    <div class="modal-actions" style="margin-top:0.55rem;justify-content:flex-start;gap:0.4rem">
      <button class="modal-action-btn primary" style="padding:0.35rem 0.7rem;font-size:0.72rem" onclick="applyCustomClusterWeights()">Apply</button>
      <button class="modal-action-btn" style="padding:0.35rem 0.7rem;font-size:0.72rem" onclick="disableCustomClusterMode()">Reset</button>
    </div>
  `;
}
function updateCustomFormulaPreview() {
  const preview = document.getElementById('custom-formula-preview');
  if (!preview) return;

  const formula = PLAYABLE
    .map(stem => `${stem}_sim × ${formatCustomWeight(customClusterWeights[stem] ?? 0)}`)
    .join(' + ');
  const weightSum = PLAYABLE.reduce((sum, stem) => sum + Math.max(0, Number(customClusterWeights[stem]) || 0), 0);
  preview.textContent = `(${formula}) / ${weightSum.toFixed(1)}`;
}
function updateCustomStemWeight(stem, value) {
  customClusterWeights[stem] = Math.max(0, Math.min(1, Number(value) || 0));
  const valueEl = document.getElementById(`custom-weight-value-${stem}`);
  if (valueEl) valueEl.textContent = formatCustomWeight(customClusterWeights[stem]);
  updateCustomFormulaPreview();
}
function updateCustomClusterButton() {
  const btn = document.getElementById('cluster-custom-btn');
  if (!btn) return;
  btn.textContent = 'Custom';
  btn.classList.toggle('active', customClusterMode);

  document.querySelectorAll('.stem-chip').forEach(chip => {
    if (chip.id === 'cluster-custom-btn') return;
    chip.classList.toggle('active', !customClusterMode && chip.dataset.stem === currentStem);
  });
}
function openCustomClusterModal() {
  const panel = document.getElementById('cluster-custom-panel');
  if (!panel) return;
  panel.innerHTML = buildCustomClusterPanelHTML();
  panel.style.display = 'block';
  updateCustomClusterButton();
}
function applyCustomClusterWeights() {
  customClusterSimilarityMatrix = buildCustomClusterSimilarityMatrix(customClusterWeights);
  customClusterMode = true;
  updateCustomClusterButton();
  renderCluster();
}
function disableCustomClusterMode() {
  customClusterMode = false;
  customClusterSimilarityMatrix = null;
  updateCustomClusterButton();
  closeCustomClusterPanel();
  renderCluster();
}
function closeCustomClusterPanel() {
  const panel = document.getElementById('cluster-custom-panel');
  if (!panel) return;
  panel.style.display = 'none';
  panel.innerHTML = '';
}
function renderClusterSizeLegend() {
  const legend = document.getElementById('cluster-size-legend');
  if (!legend) return;

  const meta = getYoutubeViewMeta();
  if (!meta) {
    legend.innerHTML = `
      <div style="font-size:0.72rem;font-weight:700;color:#e8e8f0;margin-bottom:0.25rem">Node size</div>
      <div style="font-size:0.68rem;color:var(--text-muted)">YouTube views unavailable</div>
    `;
    return;
  }

  legend.innerHTML = `
    <div style="font-size:0.72rem;font-weight:700;color:#e8e8f0;margin-bottom:0.35rem">Node size = YouTube views</div>
    <div style="display:grid;gap:0.3rem">
      ${meta.buckets.map(bucket => `
        <div style="display:flex;align-items:center;gap:0.45rem;font-size:0.68rem;color:var(--text-secondary)">
          <span style="width:${bucket.size * 2.1}px;height:${bucket.size * 2.1}px;border-radius:999px;background:rgba(124,92,252,0.9);display:inline-block;flex:0 0 auto;opacity:0.95;border:1px solid rgba(255,255,255,0.1)"></span>
          <span>${bucket.label}: ${formatYoutubeViews(bucket.min)} - ${formatYoutubeViews(bucket.max)}</span>
        </div>
      `).join('')}
    </div>
  `;
}
function simBarHTML(label, value, color, flagSilent) {
  const pct = Math.max(0, Math.min(100, value * 100));
  const silentHTML = flagSilent ? ' <span class="silent-tag">⚠ silent</span>' : '';
  return `<div class="sim-bar-container">
    <span class="sim-bar-label">${label}${silentHTML}</span>
    <div class="sim-bar-track"><div class="sim-bar-fill" style="width:${pct}%;background:${color||simColor(value)};${flagSilent?'opacity:0.3':''}"></div></div>
    <span class="sim-bar-value" style="color:${color||simColor(value)};${flagSilent?'opacity:0.3':''}">${(value*100).toFixed(1)}%</span>
  </div>`;
}
function playIcon() { return `<svg viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="5 3 19 12 5 21"/></svg>`; }
function stemPlayButtonsHTML(songIdx, songFilename, titleEscaped, { includeMix = true } = {}) {
  const mixButton = includeMix
    ? `<button class="play-btn" onclick="playAudio('${songFilename}','original','${titleEscaped}')">${playIcon()} Mix</button>`
    : '';
  const stemButtons = PLAYABLE.map(stem => {
    return `<button class="play-btn" onclick="playAudio('${songFilename}','${stem}','${titleEscaped}')">${playIcon()} ${stem}</button>`;
  }).join('');
  return `${mixButton}${stemButtons}`;
}

function stemDotsHTML(songIdx) {
  return ''; // Logic removed
}

/* ═══════════════════════════════════════════
   AUDIO CACHE & STREAMING PLAYER
   ═══════════════════════════════════════════ */
const AUDIO_CACHE = new Map(); // src -> { objectUrl, peaks, maxPeak, color }

/** Formats seconds into m:ss string, e.g. 125 → "2:05" */
function formatTime(t) {
  return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
}

function setupPlayerControls() {
  const btn = document.getElementById('play-pause-btn');
  btn.setAttribute('aria-label', 'Play / Pause');
  const audio = new Audio();
  window._audio = audio;

  btn.addEventListener('click', () => {
    if (audio.paused) { audio.play(); btn.textContent = '⏸'; }
    else { audio.pause(); btn.textContent = '▶'; }
  });

  document.getElementById('waveform-wrap').addEventListener('click', e => {
    if (!audio.duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    audio.currentTime = ((e.clientX - rect.left) / rect.width) * audio.duration;
  });

  audio.addEventListener('timeupdate', () => {
    if (!audio.duration) return;
    const pct = (audio.currentTime / audio.duration) * 100;
    document.getElementById('waveform-progress').style.width = `${pct}%`;
    document.getElementById('waveform-time').textContent =
      `${formatTime(audio.currentTime)} / ${formatTime(audio.duration)}`;
  });

  audio.addEventListener('ended', () => { btn.textContent = '▶'; });
}

// Cache of added songs keyed by filename for fast lookup
const _addedSongCache = new Map();

function getAddedSong(filename) {
    if (_addedSongCache.has(filename)) return Promise.resolve(_addedSongCache.get(filename));
    return getAllAddedSongs().then(songs => {
        songs.forEach(s => _addedSongCache.set(s.songObj.filename, s));
        return _addedSongCache.get(filename) || null;
    });
}

async function playAudio(songFilename, stem, songTitle) {
  const audio = window._audio;
  const bar = document.getElementById('player-bar');
  const songRef = DATA.songs.find(s => s.filename === songFilename) || DATA.songs.find(s => s.filename.replace('.ogg','') === songFilename);
  
  let src = '';
  if (songFilename.startsWith('ADDED_')) {
      const addedSong = await getAddedSong(songFilename);
      if (addedSong && addedSong.stems) {
          const stemKey = stem === 'original' ? 'full' : stem;
          const base64Audio = addedSong.stems[stemKey];
          if (base64Audio) {
              const binary = atob(base64Audio);
              const array = new Uint8Array(binary.length);
              for(let i=0; i<binary.length; i++) array[i] = binary.charCodeAt(i);
              const blob = new Blob([array], {type: 'audio/ogg'});
              src = URL.createObjectURL(blob);
          }
      }
      if (!src) {
          console.error("Audio data missing for uploaded track", songFilename, stem);
          return;
      }
  } else {
      const sArtist = slugify(songRef.artist_group);
      const sSong = slugify(songFilename.replace(/\.ogg$/, ''));
      const releaseBase = 'https://raw.githubusercontent.com/mynkpdr/songs-similarity-demo/audio-data/';
      
      src = stem === 'original'
        ? `${releaseBase}full--${sArtist}--${sSong}.ogg`
        : `${releaseBase}stem--${sArtist}--${sSong}--${stem}.ogg`;
  }

  // Preserve playback position if switching stems on the same song
  const keepTime = (currentPlayingFile === songFilename && !audio.paused);
  const currentTime = audio.currentTime;

  currentPlayingFile = songFilename;
  currentPlayingStem = stem;

  document.getElementById('play-pause-btn').textContent = '⏸';
  document.getElementById('np-title').textContent = songTitle || songFilename;
  document.getElementById('np-stem').textContent = stem === 'original' ? 'Full Mix' : stem;
  bar.classList.add('visible');

  // Stem buttons
  const songIdx = DATA.songs.findIndex(s => s.filename.replace('.ogg','') === songFilename);
  const btns = document.getElementById('player-stem-btns');
  btns.innerHTML = '';
  ['original', ...PLAYABLE].forEach(s => {
    const btn = document.createElement('button');
    btn.className = 'player-stem-btn' + (s === stem ? ' active' : '');
    btn.textContent = s === 'original' ? 'Mix' : s;
    btn.addEventListener('click', () => playAudio(songFilename, s, songTitle));
    btns.appendChild(btn);
  });

  // If already cached, play and render instantly (0 network overhead)
  if (AUDIO_CACHE.has(src)) {
    const cached = AUDIO_CACHE.get(src);
    audio.src = cached.objectUrl;
    if (keepTime) audio.currentTime = currentTime;
    audio.play().catch(() => {});
    drawCachedWaveform(cached);
  } else {
    // Stream in chunks, show loading waveform, then build final cache
    await fetchAndCacheAudio(src, keepTime ? currentTime : 0, songFilename, stem);
  }
}

async function fetchAndCacheAudio(src, startAtTime, songFilename, stem) {
  const canvas = document.getElementById('waveform-canvas');
  const ctx = canvas.getContext('2d');
  const wrap = document.getElementById('waveform-wrap');
  canvas.width = wrap.offsetWidth * 2;
  canvas.height = wrap.offsetHeight * 2;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const stemKey = stem === 'original' ? 'full' : stem;
  const color = STEM_COLORS[stemKey] || '#7c5cfc';
  const midY = canvas.height / 2;
  const totalBars = Math.floor(canvas.width / 3);

  try {
    if (!window.audioCtx) window.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    
    // Chunked fetching
    // Explicitly follow redirects and omit referrer for GitHub Releases CORS compatibility
    const resp = await fetch(src, {
      mode: 'cors',
      redirect: 'follow',
      referrerPolicy: 'no-referrer'
    });

    if (!resp.ok) throw new Error(`HTTP error! status: ${resp.status}`);
    const contentLength = +resp.headers.get('content-length') || 2500000;
    const reader = resp.body.getReader();
    const chunks = [];
    let received = 0;
    let lastDrawn = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;

      // Draw progressive "loading" waveform every ~100KB so it looks fast and alive
      if (received - lastDrawn > 100000 || done) {
        lastDrawn = received;
        const pct = Math.min(1, received / contentLength);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const barsToShow = Math.floor(totalBars * pct);
        for (let i = 0; i < barsToShow; i++) {
          const h = (Math.sin(i * 0.3) * 0.2 + 0.3) * canvas.height * 0.4;
          ctx.fillStyle = color + '66';
          ctx.fillRect(i * 3, midY - h/2, 2, h);
        }
        for (let i = barsToShow; i < totalBars; i++) {
          ctx.fillStyle = 'rgba(80,80,100,0.15)';
          ctx.fillRect(i * 3, midY - 1, 2, 2);
        }
      }
    }

    // Assemble file contents
    const fullBuf = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) { fullBuf.set(chunk, offset); offset += chunk.length; }
    
    // Create memory Blob for HTML5 Audio (bypassing browser double-fetching)
    const blob = new Blob([fullBuf], { type: 'audio/ogg' });
    const objectUrl = URL.createObjectURL(blob);

    // Decode to calculate peaks
    const decoded = await window.audioCtx.decodeAudioData(fullBuf.buffer);
    const raw = decoded.getChannelData(0);
    const samples = Math.floor(raw.length / totalBars);
    const peaks = [];
    for (let i = 0; i < totalBars; i++) {
      let sum = 0;
      for (let j = 0; j < samples; j++) sum += Math.abs(raw[i * samples + j]);
      peaks.push(sum / samples);
    }
    const maxPeak = Math.max(...peaks) || 1;

    // Cache everything
    const cacheData = { objectUrl, peaks, maxPeak, color };
    AUDIO_CACHE.set(src, cacheData);

    // Only set audio src and draw if user hasn't skipped to another track during download
    if (currentPlayingFile === songFilename && currentPlayingStem === stem) {
      window._audio.src = objectUrl;
      if (startAtTime > 0) window._audio.currentTime = startAtTime;
      window._audio.play().catch(()=>{});
      drawCachedWaveform(cacheData);
    }

  } catch (e) {
    console.error("Audio Load Error:", e);
    ctx.fillStyle = '#5a5a78';
    ctx.font = '20px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Waveform unavailable', canvas.width/2, canvas.height/2);
  }
}

function drawCachedWaveform(cached) {
  const canvas = document.getElementById('waveform-canvas');
  const ctx = canvas.getContext('2d');
  const wrap = document.getElementById('waveform-wrap');
  
  canvas.width = wrap.offsetWidth * 2;
  canvas.height = wrap.offsetHeight * 2;
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  const midY = height / 2;
  const bars = Math.min(cached.peaks.length, Math.floor(width / 3));

  for (let i = 0; i < bars; i++) {
    const h = (cached.peaks[i] / cached.maxPeak) * (height * 0.8);
    const x = i * 3;
    const grad = ctx.createLinearGradient(x, midY - h/2, x, midY + h/2);
    grad.addColorStop(0, cached.color + '22');
    grad.addColorStop(0.5, cached.color);
    grad.addColorStop(1, cached.color + '22');
    ctx.fillStyle = grad;
    ctx.fillRect(x, midY - h/2, 2, h);
  }
}

/* ═══════════════════════════════════════════
   MODAL POPUP
   ═══════════════════════════════════════════ */
function setupModal() {
  const overlay = document.getElementById('modal-overlay');
  document.getElementById('modal-close').addEventListener('click', closeModal);
  overlay.addEventListener('click', e => { if (e.target === overlay) closeModal(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
}
function openModal(html) {
  document.getElementById('modal-content').innerHTML = html;
  document.getElementById('modal-overlay').classList.add('open');
}
function closeModal() {
  document.getElementById('modal-overlay').classList.remove('open');
}

function buildSongModalHTML(songIdx, context) {
  const song = DATA.songs[songIdx];
  const fn = song.filename.replace('.ogg','');

  // Top 5 similar by current stem
  const sims = DATA.similarities[currentStem][songIdx];
  const top5 = sims.map((v,i)=>({i,v})).filter(d=>d.i!==songIdx).sort((a,b)=>b.v-a.v).slice(0,5);

  let stemRows = '';
  // Removed Stem Energy calculation as it is not consistently available for user-added songs

  return `
    <h2>${songName(song)}</h2>
    <div class="modal-sub">${song.film || ''} ${song.year ? '('+song.year+')' : ''} · ${song.singer || ''} · ${song.composer || ''}</div>
    ${song.primary_mood ? `<div style="margin-bottom:0.75rem"><span class="mood-badge">${song.primary_mood}</span> ${song.secondary_moods && song.secondary_moods !== 'nan' ? `<span style="font-size:0.7rem;color:var(--text-muted);margin-left:0.3rem">${song.secondary_moods}</span>` : ''}</div>` : ''}
    ${song.youtube_url && song.youtube_url !== 'nan' ? `<div style="margin-bottom:0.75rem"><a href="${song.youtube_url}" target="_blank" rel="noopener" style="display:inline-flex;align-items:center;gap:0.4rem;padding:0.35rem 0.8rem;background:#ff0000;color:#fff;border-radius:6px;font-size:0.78rem;font-weight:600;text-decoration:none;transition:opacity 0.2s" onmouseover="this.style.opacity=0.85" onmouseout="this.style.opacity=1">▶ Watch on YouTube</a></div>` : ''}

    <div class="modal-section">
      <h4>🎧 Listen</h4>
      <div class="modal-actions">
        <button class="play-btn" onclick="playAudio('${fn}','original','${esc(songName(song))}')">${playIcon()} Full Mix</button>
        ${PLAYABLE.map(s => {
          return `<button class="play-btn" onclick="playAudio('${fn}','${s}','${esc(songName(song))}')">${playIcon()} ${s}</button>`;
        }).join('')}
      </div>
    </div>

    <div class="modal-section">
      <h4>🔗 Most Similar (by ${currentStem})</h4>
      ${top5.map((r,rank) => {
        const s2 = DATA.songs[r.i];
        return `<div style="display:flex;align-items:center;gap:0.5rem;padding:0.35rem 0;border-bottom:1px solid var(--border)">
          <span style="font-family:'JetBrains Mono';font-size:0.7rem;color:var(--text-muted);min-width:1.5rem">#${rank+1}</span>
          <span style="flex:1;font-size:0.82rem;font-weight:500">${songName(s2)}</span>
          <span style="font-family:'JetBrains Mono';font-size:0.78rem;font-weight:600;color:${simColor(r.v)}">${(r.v*100).toFixed(1)}%</span>
        </div>`;
      }).join('')}
    </div>

    <div class="modal-actions" style="margin-top:1rem">
      <button class="modal-action-btn primary" onclick="closeModal();selectedSongIdx=${songIdx};document.querySelector('[data-tab=similarity]').click();renderSimilarityResults(${songIdx});highlightSidebarItem(${songIdx})">View Full Similarity</button>
      <button class="modal-action-btn" onclick="closeModal();openCompareWithSongs([${songIdx}], { keepFocus: false })">Compare Song</button>
    </div>
  `;
}

function buildPairModalHTML(i, j) {
  const sA = DATA.songs[i], sB = DATA.songs[j];
  const fnA = sA.filename.replace('.ogg',''), fnB = sB.filename.replace('.ogg','');

  let stemRows = '';
  ALL_STEMS.forEach(stem => {
    const val = DATA.similarities[stem]?.[i]?.[j] ?? 0;
    stemRows += simBarHTML(stem, val, STEM_COLORS[stem]);
  });

  return `
    <h2 style="font-size:0.95rem">${songName(sA)} <span style="color:var(--text-muted);font-weight:400">×</span> ${songName(sB)}</h2>
    <div class="modal-sub">${sA.film||''} vs ${sB.film||''}</div>

    <div style="text-align:center;margin:1rem 0">
      <div style="font-size:2.5rem;font-weight:900;background:linear-gradient(135deg,var(--accent-1),var(--accent-2));-webkit-background-clip:text;-webkit-text-fill-color:transparent">
        ${((DATA.similarities.full?.[i]?.[j]??0)*100).toFixed(1)}%
      </div>
      <div style="font-size:0.75rem;color:var(--text-secondary)">Full Mix Similarity</div>
    </div>

    <div class="modal-section">
      <h4>Per-Stem Breakdown</h4>
      ${stemRows}
    </div>

    <div class="modal-section">
      <h4>🎧 Listen & Compare</h4>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem">
        <div>
          <div style="font-size:0.72rem;color:var(--text-secondary);margin-bottom:0.3rem">${songName(sA)}</div>
          <div style="display:flex;gap:0.35rem;flex-wrap:wrap">${stemPlayButtonsHTML(i, fnA, esc(songName(sA)), { includeMix: true })}</div>
        </div>
        <div>
          <div style="font-size:0.72rem;color:var(--text-secondary);margin-bottom:0.3rem">${songName(sB)}</div>
          <div style="display:flex;gap:0.35rem;flex-wrap:wrap">${stemPlayButtonsHTML(j, fnB, esc(songName(sB)), { includeMix: true })}</div>
        </div>
      </div>
    </div>

    <div class="modal-actions" style="margin-top:1rem">
      <button class="modal-action-btn primary" onclick="closeModal();openCompareWithSongs([${i},${j}])">Full Comparison</button>
    </div>
  `;
}



/* ═══════════════════════════════════════════
   PANEL 2: SIMILARITY
   ═══════════════════════════════════════════ */
function renderSimilarityList() {
  const list = document.getElementById('sim-song-list');
  const search = document.getElementById('sim-search');
  function render(q) {
    list.innerHTML = '';
    DATA.songs.forEach((song, idx) => {
      const name = songName(song);
      if (q && !name.toLowerCase().includes(q.toLowerCase())) return;
      const item = document.createElement('div');
      item.className = 'sim-song-item' + (idx === selectedSongIdx ? ' active' : '');
      item.dataset.idx = idx;
      item.innerHTML = `<span class="idx">${idx+1}</span><span class="name">${name}</span>`;
      item.addEventListener('click', () => { 
        selectedSongIdx = idx; 
        updateURLState();
        renderSimilarityResults(idx); 
        highlightSidebarItem(idx); 
      });
      list.appendChild(item);
    });
  }
  render('');
  search.addEventListener('input', debounce(() => render(search.value), 150));
}

function highlightSidebarItem(idx) {
  document.querySelectorAll('.sim-song-item').forEach(el =>
    el.classList.toggle('active', parseInt(el.dataset.idx) === idx)
  );
}

function renderSimilarityResults(idx) {
  const song = DATA.songs[idx];
  const fn = song.filename.replace('.ogg','');

  const header = document.getElementById('sim-header');
  header.innerHTML = `
    <h2>${songName(song)}</h2>
    <div class="sub">
      ${song.film || ''} ${song.year ? '('+song.year+')' : ''} — ${song.singer || ''}
      | Stem: <strong style="color:${STEM_COLORS[currentStem]}">${currentStem}</strong>
    </div>
    <div style="margin-top:0.5rem;display:flex;gap:0.3rem;flex-wrap:wrap">
      ${stemPlayButtonsHTML(idx, fn, esc(songName(song)))}
    </div>
  `;

  const sims = DATA.similarities[currentStem][idx];
  const ranked = sims.map((v,i)=>({idx:i,sim:v})).filter(d=>d.idx!==idx).sort((a,b)=>b.sim-a.sim).slice(0, 50);

  const details = document.getElementById('sim-details');
  details.innerHTML = '';
  const gridEl = document.createElement('div');
  gridEl.className = 'sim-results-grid';

  ranked.forEach((r, rank) => {
    const s = DATA.songs[r.idx];
    const card = document.createElement('div');
    card.className = 'sim-result-card';

    let stemBars = '';
    ALL_STEMS.forEach(stem => {
      const val = DATA.similarities[stem]?.[idx]?.[r.idx] ?? 0;
      stemBars += simBarHTML(stem, val, STEM_COLORS[stem]);
    });

    card.innerHTML = `
      <div class="rank">#${rank+1}</div>
      <div class="res-title">${songName(s)}</div>
      <div class="res-meta">${s.film||''} ${s.year?'('+s.year+')':''}</div>
      <div style="margin-bottom:0.4rem">
        <div class="sim-bar-container">
          <span class="sim-bar-label" style="font-weight:600">Score</span>
          <div class="sim-bar-track"><div class="sim-bar-fill" style="width:${r.sim*100}%;background:${simColor(r.sim)}"></div></div>
          <span class="sim-bar-value" style="color:${simColor(r.sim)};font-size:0.82rem">${(r.sim*100).toFixed(1)}%</span>
        </div>
      </div>
      ${stemBars}
      ${stemDotsHTML(r.idx)}
      <div style="margin-top:0.4rem;display:flex;gap:0.25rem;flex-wrap:wrap">
        <button class="play-btn" onclick="event.stopPropagation();playAudio('${s.filename.replace('.ogg','')}','original','${esc(songName(s))}')">${playIcon()} Mix</button>
        ${PLAYABLE.map(stem => {
          return `<button class="play-btn" onclick="event.stopPropagation();playAudio('${s.filename.replace('.ogg','')}','${stem}','${esc(songName(s))}')">${playIcon()} ${stem}</button>`;
        }).join('')}
      </div>
    `;
    card.addEventListener('click', () => { 
      selectedSongIdx = r.idx; 
      renderSimilarityResults(r.idx); 
      highlightSidebarItem(r.idx); 
      updateURLState();
    });
    gridEl.appendChild(card);
  });
  details.appendChild(gridEl);
}

/* ═══════════════════════════════════════════
   PANEL 3: COMPARE
   ═══════════════════════════════════════════ */
function buildCompareSelects() {
  const addSelect = document.getElementById('compare-add-select');
  if (!addSelect) return;
  buildCompareStemBar();

  addSelect.innerHTML = '';
  DATA.songs.forEach((song, idx) => addSelect.add(new Option(songName(song), idx)));

  const addBtn = document.getElementById('compare-add-btn');
  if (addBtn && !addBtn.dataset.bound) {
    addBtn.dataset.bound = '1';
    addBtn.addEventListener('click', addCompareSong);
  }

  const resetBtn = document.getElementById('compare-reset-btn');
  if (resetBtn && !resetBtn.dataset.bound) {
    resetBtn.dataset.bound = '1';
    resetBtn.addEventListener('click', resetCompareSelection);
  }

  if (!addSelect.dataset.bound) {
    addSelect.dataset.bound = '1';
    addSelect.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter') return;
      event.preventDefault();
      addCompareSong();
    });
  }

  compareSelectionNotice = '';
  compareSelection = sanitizeCompareSelection([], { fillDefaults: true });
  compareFocusedPair = chooseCompareFocusedPair(compareSelection);
  renderCompare();
}

function buildCompareStemBar() {
  const bar = document.getElementById('compare-stems');
  if (!bar) return;
  bar.innerHTML = '';
  ALL_STEMS.forEach((stem) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = `compare-stem-chip${stem === compareMatrixStem ? ' active' : ''}`;
    chip.dataset.stem = stem;
    chip.textContent = stem;
    chip.addEventListener('click', () => setCompareMatrixStem(stem));
    bar.appendChild(chip);
  });
}

function setCompareMatrixStem(stem) {
  if (!ALL_STEMS.includes(stem)) return;
  compareMatrixStem = stem;
  document.querySelectorAll('#compare-stems .compare-stem-chip').forEach((chip) => {
    chip.classList.toggle('active', chip.dataset.stem === stem);
  });
  renderCompare();
}

function sanitizeCompareSelection(songIds, { fillDefaults = true, capToMax = true } = {}) {
  const songCount = DATA?.songs?.length ?? 0;
  if (!songCount) return [];

  const unique = [];
  (songIds || []).forEach((raw) => {
    const idx = Number(raw);
    if (!Number.isInteger(idx) || idx < 0 || idx >= songCount) return;
    if (unique.includes(idx)) return;
    unique.push(idx);
  });

  const minNeeded = Math.min(2, songCount);
  if (fillDefaults && unique.length < minNeeded) {
    for (let i = 0; i < songCount && unique.length < minNeeded; i++) {
      if (!unique.includes(i)) unique.push(i);
    }
  }

  if (capToMax && unique.length > MAX_COMPARE_SONGS) {
    return unique.slice(0, MAX_COMPARE_SONGS);
  }
  return unique;
}

function chooseCompareFocusedPair(songIds) {
  if (!songIds || songIds.length < 2) return null;
  let bestPair = [songIds[0], songIds[1]];
  let bestScore = -1;
  for (let i = 0; i < songIds.length; i++) {
    for (let j = i + 1; j < songIds.length; j++) {
      const a = songIds[i];
      const b = songIds[j];
      const sim = getSimilarity(compareMatrixStem, a, b);
      if (sim > bestScore) {
        bestScore = sim;
        bestPair = [a, b];
      }
    }
  }
  return bestPair;
}

function isComparePairValid(pair, selection) {
  if (!pair || pair.length !== 2) return false;
  const [a, b] = pair;
  return a !== b && selection.includes(a) && selection.includes(b);
}

function rankSongIdsByCohesion(songIds, simsMatrix) {
  const matrix = simsMatrix || DATA?.similarities?.full;
  const uniqueIds = sanitizeCompareSelection(songIds, { fillDefaults: false, capToMax: false });
  if (uniqueIds.length <= 1) return uniqueIds;

  const getPair = (a, b) => {
    if (matrix?.[a] && Number.isFinite(matrix[a][b])) return matrix[a][b];
    if (matrix?.[b] && Number.isFinite(matrix[b][a])) return matrix[b][a];
    return getSimilarity('full', a, b);
  };

  return uniqueIds
    .map((id) => {
      const neighbors = uniqueIds.filter(other => other !== id);
      const avg = neighbors.reduce((sum, other) => sum + getPair(id, other), 0) / neighbors.length;
      return { id, avg };
    })
    .sort((a, b) => {
      const diff = b.avg - a.avg;
      if (Math.abs(diff) > 1e-9) return diff;
      return songName(DATA.songs[a.id]).localeCompare(songName(DATA.songs[b.id]));
    })
    .map(item => item.id);
}

function setCompareSelection(songIds, { keepFocus = false, notice = '' } = {}) {
  compareSelection = sanitizeCompareSelection(songIds, { fillDefaults: true, capToMax: true });
  compareSelectionNotice = notice || '';
  if (!keepFocus || !isComparePairValid(compareFocusedPair, compareSelection)) {
    compareFocusedPair = chooseCompareFocusedPair(compareSelection);
  }
}

function openCompareWithSongs(songIds, options = {}) {
  setCompareSelection(songIds, {
    keepFocus: !!options.keepFocus,
    notice: options.notice || ''
  });
  const tabBtn = document.querySelector('.tab-btn[data-tab="compare"]');
  if (tabBtn) tabBtn.click();
  renderCompare();
}

function compareClusterSelection(songIds) {
  const allIds = sanitizeCompareSelection(songIds, { fillDefaults: false, capToMax: false });
  if (allIds.length < 2) return;

  const ranked = rankSongIdsByCohesion(allIds, getClusterSimilarityMatrix());
  const compareIds = ranked.slice(0, MAX_COMPARE_SONGS);
  const notice = allIds.length > MAX_COMPARE_SONGS
    ? `Selected ${allIds.length} songs. Showing top ${MAX_COMPARE_SONGS} by in-selection cohesion.`
    : '';
  openCompareWithSongs(compareIds, { notice });
}

function addCompareSong() {
  const addSelect = document.getElementById('compare-add-select');
  if (!addSelect) return;
  const songIdx = Number(addSelect.value);
  if (!Number.isInteger(songIdx)) return;
  if (compareSelection.includes(songIdx)) {
    compareSelectionNotice = `${songName(DATA.songs[songIdx])} is already selected.`;
    renderCompare();
    return;
  }
  if (compareSelection.length >= MAX_COMPARE_SONGS) {
    compareSelectionNotice = `You can compare up to ${MAX_COMPARE_SONGS} songs at once.`;
    renderCompare();
    return;
  }
  setCompareSelection([...compareSelection, songIdx], { keepFocus: true });
  renderCompare();
}

function removeCompareSong(songIdx) {
  const minNeeded = Math.min(2, DATA?.songs?.length ?? 0);
  if (compareSelection.length <= minNeeded) {
    compareSelectionNotice = `At least ${minNeeded} songs are required for comparison.`;
    renderCompare();
    return;
  }
  setCompareSelection(compareSelection.filter(idx => idx !== Number(songIdx)), { keepFocus: false });
  renderCompare();
}

function resetCompareSelection() {
  compareSelectionNotice = '';
  compareFocusedPair = null;
  compareSelection = sanitizeCompareSelection([], { fillDefaults: true, capToMax: true });
  compareFocusedPair = chooseCompareFocusedPair(compareSelection);
  renderCompare();
}

function setCompareFocusedPair(idxA, idxB) {
  compareFocusedPair = [Number(idxA), Number(idxB)];
  renderCompare();
}

function renderCompareSelectionChips() {
  const list = document.getElementById('compare-selected-list');
  const addSelect = document.getElementById('compare-add-select');
  const addBtn = document.getElementById('compare-add-btn');
  const note = document.getElementById('compare-limit-note');
  if (!list || !addSelect || !addBtn || !note) return;

  const minNeeded = Math.min(2, DATA?.songs?.length ?? 0);
  const disableRemove = compareSelection.length <= minNeeded;
  const focusSet = new Set(compareFocusedPair || []);

  list.innerHTML = compareSelection.map((songIdx, rank) => {
    const song = DATA.songs[songIdx];
    const fn = song.filename.replace('.ogg', '');
    const inFocus = focusSet.has(songIdx);
    return `
      <div class="compare-song-pill${inFocus ? ' in-focus' : ''}">
        <div class="compare-song-pill-head">
          <span class="compare-song-pill-rank">${rank + 1}</span>
          <span class="compare-song-pill-name">${songName(song)}</span>
        </div>
        <div class="compare-song-pill-actions">
          <button class="play-btn" onclick="playAudio('${fn}','original','${esc(songName(song))}')">${playIcon()} Mix</button>
          <button class="compare-chip-remove" ${disableRemove ? 'disabled' : ''} onclick="removeCompareSong(${songIdx})">✕</button>
        </div>
      </div>
    `;
  }).join('');

  const availableIndex = DATA.songs.findIndex((_, idx) => !compareSelection.includes(idx));
  if (availableIndex >= 0) addSelect.value = String(availableIndex);

  const atMax = compareSelection.length >= MAX_COMPARE_SONGS || availableIndex < 0;
  addSelect.disabled = atMax;
  addBtn.disabled = atMax;

  const defaultNote = atMax
    ? `Maximum ${MAX_COMPARE_SONGS} songs selected.`
    : 'Tip: select songs from Cluster view and compare them in one click.';
  note.textContent = compareSelectionNotice || defaultNote;
}

function getComparePairStats(selection, stem = 'full') {
  const pairs = [];
  for (let i = 0; i < selection.length; i++) {
    for (let j = i + 1; j < selection.length; j++) {
      const a = selection[i];
      const b = selection[j];
      pairs.push({ a, b, sim: getSimilarity(stem, a, b) });
    }
  }

  const pairCount = pairs.length;
  const avgStem = pairCount
    ? pairs.reduce((sum, pair) => sum + pair.sim, 0) / pairCount
    : 0;

  const strongestPair = pairCount
    ? pairs.reduce((best, pair) => pair.sim > best.sim ? pair : best, pairs[0])
    : null;

  const weakestPair = pairCount
    ? pairs.reduce((worst, pair) => pair.sim < worst.sim ? pair : worst, pairs[0])
    : null;

  const cohesionBySong = selection.map((songIdx) => {
    const neighbors = selection.filter(other => other !== songIdx);
    if (!neighbors.length) return { songIdx, avg: 0 };
    const avg = neighbors.reduce((sum, other) => sum + getSimilarity(stem, songIdx, other), 0) / neighbors.length;
    return { songIdx, avg };
  }).sort((a, b) => b.avg - a.avg);

  return {
    stem,
    selection,
    pairCount,
    pairs,
    avgStem,
    strongestPair,
    weakestPair,
    representative: cohesionBySong[0] || null
  };
}

function renderCompareSummary(stats) {
  const summary = document.getElementById('compare-summary-grid');
  if (!summary) return;
  if (!stats?.pairCount) {
    summary.innerHTML = '<div class="compare-empty-state">Pick at least 2 songs to view group summary.</div>';
    return;
  }

  const strongLabel = `${songName(DATA.songs[stats.strongestPair.a])} x ${songName(DATA.songs[stats.strongestPair.b])}`;
  const weakLabel = `${songName(DATA.songs[stats.weakestPair.a])} x ${songName(DATA.songs[stats.weakestPair.b])}`;
  const repr = stats.representative;
  const stemLabel = stats.stem === 'full' ? 'full-mix' : `${stats.stem}`;

  summary.innerHTML = `
    <div class="compare-stat-card">
      <div class="compare-stat-label">Group Cohesion</div>
      <div class="compare-stat-value" style="color:${simColor(stats.avgStem)}">${(stats.avgStem * 100).toFixed(1)}%</div>
      <div class="compare-stat-sub">Average ${stemLabel} similarity across all pairs</div>
    </div>
    <div class="compare-stat-card">
      <div class="compare-stat-label">Strongest Pair</div>
      <div class="compare-stat-value" style="color:${simColor(stats.strongestPair.sim)}">${(stats.strongestPair.sim * 100).toFixed(1)}%</div>
      <div class="compare-stat-sub">${strongLabel}</div>
    </div>
    <div class="compare-stat-card">
      <div class="compare-stat-label">Most Distinct Pair</div>
      <div class="compare-stat-value" style="color:${simColor(stats.weakestPair.sim)}">${(stats.weakestPair.sim * 100).toFixed(1)}%</div>
      <div class="compare-stat-sub">${weakLabel}</div>
    </div>
    <div class="compare-stat-card">
      <div class="compare-stat-label">Representative Song</div>
      <div class="compare-stat-value" style="font-size:1.2rem;color:var(--text-primary)">${repr ? songName(DATA.songs[repr.songIdx]) : '—'}</div>
      <div class="compare-stat-sub">Highest average similarity inside selection</div>
    </div>
  `;
}

function renderCompareMatrix(stats) {
  const matrix = document.getElementById('compare-matrix');
  const title = document.getElementById('compare-matrix-title');
  const sub = document.getElementById('compare-matrix-sub');
  if (!matrix) return;

  const stemTitle = compareMatrixStem === 'full'
    ? 'Pairwise Similarity (Full Mix)'
    : `Pairwise Similarity (${compareMatrixStem[0].toUpperCase()}${compareMatrixStem.slice(1)} Stem)`;
  if (title) title.textContent = stemTitle;
  if (sub) sub.textContent = 'Matrix view inspired by analytics products: fast scanning, click-to-focus details.';

  if (!stats?.pairCount) {
    matrix.innerHTML = '<div class="compare-empty-state">Need at least 2 songs for matrix view.</div>';
    return;
  }

  const selected = stats.selection;
  let gridHTML = '<div class="compare-matrix-grid">';
  gridHTML += '<div class="compare-matrix-cell compare-matrix-corner">Song</div>';
  selected.forEach((songIdx, col) => {
    gridHTML += `<div class="compare-matrix-cell compare-matrix-head">${col + 1}. ${songName(DATA.songs[songIdx])}</div>`;
  });

  selected.forEach((rowIdx, row) => {
    gridHTML += `<div class="compare-matrix-cell compare-matrix-row">${row + 1}. ${songName(DATA.songs[rowIdx])}</div>`;
    selected.forEach((colIdx) => {
      if (rowIdx === colIdx) {
        gridHTML += '<div class="compare-matrix-cell compare-matrix-diag">—</div>';
        return;
      }
      const sim = getSimilarity(compareMatrixStem, rowIdx, colIdx);
      const isFocused = !!compareFocusedPair &&
        compareFocusedPair.includes(rowIdx) &&
        compareFocusedPair.includes(colIdx);
      const alpha = (0.14 + sim * 0.46).toFixed(3);
      gridHTML += `
        <button
          class="compare-matrix-cell compare-matrix-value${isFocused ? ' active' : ''}"
          style="--sim-color:${simColor(sim)};--sim-alpha:${alpha}"
          onclick="setCompareFocusedPair(${rowIdx},${colIdx})"
        >${(sim * 100).toFixed(1)}%</button>
      `;
    });
  });
  gridHTML += '</div>';

  matrix.innerHTML = `<div class="compare-matrix-scroll" style="--compare-count:${selected.length}">${gridHTML}</div>`;
}

function renderComparePairDetail(stats) {
  const detail = document.getElementById('compare-pair-detail');
  if (!detail) return;
  if (!stats?.pairCount || !isComparePairValid(compareFocusedPair, stats.selection)) {
    detail.innerHTML = '<div class="compare-empty-state">Click a matrix cell to inspect a pair.</div>';
    return;
  }

  const [idxA, idxB] = compareFocusedPair;
  const songA = DATA.songs[idxA];
  const songB = DATA.songs[idxB];
  const fnA = songA.filename.replace('.ogg', '');
  const fnB = songB.filename.replace('.ogg', '');
  const selectedStemSim = getSimilarity(compareMatrixStem, idxA, idxB);
  const scoreLabel = compareMatrixStem === 'full'
    ? 'Full Mix Similarity'
    : `${compareMatrixStem[0].toUpperCase()}${compareMatrixStem.slice(1)} Stem Similarity`;

  const stemRows = ALL_STEMS.map((stem) => {
    const sim = getSimilarity(stem, idxA, idxB);
    return `
      <div class="compare-pair-stem-row">
        <div class="compare-pair-stem-name" style="color:${STEM_COLORS[stem]}">${stem}</div>
        <div class="compare-pair-stem-track">
          <div class="compare-pair-stem-fill" style="width:${(sim * 100).toFixed(1)}%;background:${STEM_COLORS[stem]}"></div>
        </div>
        <div class="compare-pair-stem-val" style="color:${STEM_COLORS[stem]}">${(sim * 100).toFixed(1)}%</div>
      </div>
    `;
  }).join('');

  const activeStemVals = PLAYABLE.map(stem => ({
    stem,
    sim: getSimilarity(stem, idxA, idxB),
    active: true
  }));

  let insight = 'Stem activity is limited for this pair, so stem-level interpretation may be unstable.';
  if (activeStemVals.length) {
    const maxStem = activeStemVals.reduce((a, b) => a.sim > b.sim ? a : b, activeStemVals[0]);
    const minStem = activeStemVals.reduce((a, b) => a.sim < b.sim ? a : b, activeStemVals[0]);
    insight = `Most aligned stem: <strong style="color:${STEM_COLORS[maxStem.stem]}">${maxStem.stem}</strong> (${(maxStem.sim * 100).toFixed(1)}%). ` +
      `Most contrasting stem: <strong style="color:${STEM_COLORS[minStem.stem]}">${minStem.stem}</strong> (${(minStem.sim * 100).toFixed(1)}%).`;
  }

  detail.innerHTML = `
    <div class="compare-pair-head">
      <h3>${songName(songA)} × ${songName(songB)}</h3>
      <div class="compare-pair-score" style="color:${simColor(selectedStemSim)}">${(selectedStemSim * 100).toFixed(1)}%</div>
      <div class="compare-pair-score-label">${scoreLabel}</div>
    </div>

    <div class="compare-pair-song-grid">
      <div class="compare-pair-song">
        <div class="compare-pair-song-name">${songName(songA)}</div>
        <div class="compare-pair-song-meta">${songA.film || '—'} ${songA.year ? `(${songA.year})` : ''}</div>
        <div style="display:flex;gap:0.35rem;flex-wrap:wrap">
          ${stemPlayButtonsHTML(idxA, fnA, esc(songName(songA)), { includeMix: true })}
        </div>
      </div>
      <div class="compare-pair-song">
        <div class="compare-pair-song-name">${songName(songB)}</div>
        <div class="compare-pair-song-meta">${songB.film || '—'} ${songB.year ? `(${songB.year})` : ''}</div>
        <div style="display:flex;gap:0.35rem;flex-wrap:wrap">
          ${stemPlayButtonsHTML(idxB, fnB, esc(songName(songB)), { includeMix: true })}
        </div>
      </div>
    </div>

    <div class="compare-pair-stems">${stemRows}</div>

    <div class="stem-contrib-section" style="margin-top:0.75rem">
      <h3>Pair Insight</h3>
      <p style="font-size:0.78rem;color:var(--text-secondary);line-height:1.6">${insight}</p>
    </div>

    <div class="modal-actions" style="margin-top:0.75rem">
      <button class="modal-action-btn" onclick="openModal(buildPairModalHTML(${idxA},${idxB}))">Open Pair Deep Dive</button>
    </div>
  `;
}

function renderCompare() {
  if (!DATA?.songs?.length) return;

  buildCompareStemBar();
  compareSelection = sanitizeCompareSelection(compareSelection, { fillDefaults: true, capToMax: true });
  if (!isComparePairValid(compareFocusedPair, compareSelection)) {
    compareFocusedPair = chooseCompareFocusedPair(compareSelection);
  }

  renderCompareSelectionChips();
  const stats = getComparePairStats(compareSelection, compareMatrixStem);
  renderCompareSummary(stats);
  renderCompareMatrix(stats);
  renderComparePairDetail(stats);
}

/* ═══════════════════════════════════════════
   PANEL 4: HEATMAP (no axis labels, fit-to-container, right-panel detail)
   ═══════════════════════════════════════════ */
let heatmapLocked = false, heatmapLockedI = -1, heatmapLockedJ = -1;

function renderHeatmap() {
  const container = document.getElementById('heatmap-container');
  container.innerHTML = '';
  container.style.overflow = 'auto';
  container.style.alignItems = 'flex-start';
  container.style.justifyContent = 'flex-start';
  heatmapLocked = false;
  const detail = document.getElementById('heatmap-detail');
  detail.classList.remove('locked');
  detail.innerHTML = '<div class="heatmap-detail-empty">Hover a cell to preview<br>Click to lock & interact</div>';

  const sims = DATA.similarities[currentStem];
  if (!sims) return;
  const n = DATA.songs.length;

  const offDiag = [];
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      if (i !== j) offDiag.push(sims[i][j]);
  offDiag.sort((a, b) => a - b);
  const p = (arr, pct) => arr[Math.floor(arr.length * pct / 100)];
  const vMin = p(offDiag, 2);
  const vMax = p(offDiag, 98);
  const vMid = (vMin + vMax) / 2;
  const vQ1 = (vMin + vMid) / 2;
  const vQ3 = (vMid + vMax) / 2;

  const box = container.getBoundingClientRect();
  const legendH = 44;
  const minCellSize = n > 120 ? 9 : n > 80 ? 11 : 14;
  const maxCellSize = n > 120 ? 12 : n > 80 ? 14 : 18;
  const targetCellSize = Math.floor(((box.width || 600) - 24) / Math.max(1, Math.min(n, 60)));
  const cellSize = Math.max(minCellSize, Math.min(maxCellSize, targetCellSize));
  const totalSize = cellSize * n;
  const pad = 10;
  const W = totalSize + pad * 2, H = totalSize + pad * 2 + legendH;

  const wrap = document.createElement('div');
  wrap.style.cssText = `position:relative;width:${W}px;height:${H}px;min-width:${W}px;min-height:${H}px;margin:0;flex:0 0 auto;overflow:auto`;
  container.appendChild(wrap);

  const canvas = document.createElement('canvas');
  canvas.width = W; canvas.height = H;
  canvas.style.cssText = 'position:absolute;top:0;left:0;display:block;';
  wrap.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svgEl.setAttribute('width', W);
  svgEl.setAttribute('height', H);
  svgEl.style.cssText = 'position:absolute;top:0;left:0;cursor:pointer;';
  wrap.appendChild(svgEl);
  const svg = d3.select(svgEl);

  const colorScale = d3.scaleLinear()
    .domain([vMin, vQ1, vMid, vQ3, vMax])
    .range(['#0d1b2a', '#1b4965', '#62b6cb', '#f4845f', '#f72585'])
    .clamp(true);

  const radius = cellSize >= 12 ? 2 : 1;
  ctx.clearRect(0, 0, W, H);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const val = sims[i][j];
      ctx.fillStyle = i === j ? 'rgba(124,92,252,0.4)' : colorScale(val);
      ctx.globalAlpha = 0.95;
      
      const x = pad + j * cellSize;
      const y = pad + i * cellSize;
      const w = cellSize - 1;
      
      ctx.beginPath();
      if (ctx.roundRect) {
         ctx.roundRect(x, y, w, w, radius);
      } else {
         ctx.rect(x, y, w, w);
      }
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1.0;

  const highlight = svg.append('rect')
    .attr('fill', 'none').attr('stroke', '#fff').attr('stroke-width', 2).attr('rx', radius)
    .style('display', 'none').style('pointer-events', 'none');

  svgEl.addEventListener('mousemove', e => {
    const rect = svgEl.getBoundingClientRect();
    const x = e.clientX - rect.left - pad;
    const y = e.clientY - rect.top - pad;
    const j = Math.floor(x / cellSize);
    const i = Math.floor(y / cellSize);
    
    if (i >= 0 && i < n && j >= 0 && j < n) {
      highlight.style('display', 'block')
        .attr('x', pad + j * cellSize).attr('y', pad + i * cellSize)
        .attr('width', cellSize - 1).attr('height', cellSize - 1);
      if (!heatmapLocked) showHeatmapDetail(i, j, false);
    } else {
      if (!heatmapLocked) {
        highlight.style('display', 'none');
        detail.innerHTML = '<div class="heatmap-detail-empty">Hover a cell to preview<br>Click to lock & interact</div>';
      }
    }
  });

  svgEl.addEventListener('mouseout', () => {
    if (!heatmapLocked) {
      highlight.style('display', 'none');
      detail.innerHTML = '<div class="heatmap-detail-empty">Hover a cell to preview<br>Click to lock & interact</div>';
    }
  });

  svgEl.addEventListener('click', e => {
    const rect = svgEl.getBoundingClientRect();
    const x = e.clientX - rect.left - pad;
    const y = e.clientY - rect.top - pad;
    const j = Math.floor(x / cellSize);
    const i = Math.floor(y / cellSize);
    if (i >= 0 && i < n && j >= 0 && j < n) {
      heatmapLocked = true;
      heatmapLockedI = i;
      heatmapLockedJ = j;
      showHeatmapDetail(i, j, true);
    }
  });

  // ── Legend bar ──
  const lgY = totalSize + pad * 2 + 6;
  const lgX = pad;
  const lgW = totalSize;
  const lgH = 10;
  const defs = svg.append('defs');
  const gradId = 'hm-legend-grad-' + currentStem;
  const grad = defs.append('linearGradient').attr('id', gradId);
  [0, 0.25, 0.5, 0.75, 1.0].forEach(t => {
    const v = vMin + t * (vMax - vMin);
    grad.append('stop').attr('offset', (t * 100) + '%').attr('stop-color', colorScale(v));
  });
  svg.append('rect').attr('x', lgX).attr('y', lgY).attr('width', lgW).attr('height', lgH)
    .attr('rx', 4).attr('fill', `url(#${gradId})`);
  // Tick labels
  const ticks = [vMin, vQ1, vMid, vQ3, vMax];
  ticks.forEach((v, idx) => {
    const tx = lgX + (idx / (ticks.length - 1)) * lgW;
    svg.append('text').attr('x', tx).attr('y', lgY + lgH + 12)
      .attr('text-anchor', 'middle').attr('fill', '#8888a8')
      .attr('font-size', '8px').attr('font-family', "'JetBrains Mono', monospace")
      .text((v * 100).toFixed(0) + '%');
  });
  // Label
  svg.append('text').attr('x', lgX + lgW / 2).attr('y', lgY - 4)
    .attr('text-anchor', 'middle').attr('fill', '#e8e8f0')
    .attr('font-size', '9px').attr('font-weight', '600')
    .text(`${currentStem} similarity`);
}

function showHeatmapDetail(i, j, locked) {
  const detail = document.getElementById('heatmap-detail');
  detail.classList.toggle('locked', locked);
  const sA = DATA.songs[i], sB = DATA.songs[j];
  const fnA = sA.filename.replace('.ogg',''), fnB = sB.filename.replace('.ogg','');
  const fullSim = DATA.similarities.full?.[i]?.[j] ?? 0;

  let stemBars = '';
  ALL_STEMS.forEach(stem => {
    const val = DATA.similarities[stem]?.[i]?.[j] ?? 0;
    stemBars += simBarHTML(stem, val, STEM_COLORS[stem]);
  });

  detail.innerHTML = `
    <div class="heatmap-detail-header">
      <h3>${songName(sA)}</h3>
      <div class="sub">${sA.film||''} · ${sA.singer||''}</div>
      <div style="margin:0.3rem 0;font-size:0.72rem;color:var(--text-muted)">×</div>
      <h3>${songName(sB)}</h3>
      <div class="sub">${sB.film||''} · ${sB.singer||''}</div>
    </div>
    <div class="heatmap-detail-score">
      <div class="big">${(fullSim*100).toFixed(1)}%</div>
      <div class="label">Full Mix Similarity</div>
    </div>
    ${stemBars}
    ${locked ? `
      <div style="margin-top:0.75rem">
        <div style="font-size:0.72rem;color:var(--text-secondary);margin-bottom:0.3rem">${songName(sA)}</div>
        <div style="display:flex;gap:0.25rem;flex-wrap:wrap;margin-bottom:0.5rem">
          <button class="play-btn" onclick="playAudio('${fnA}','original','${esc(songName(sA))}')">▶ Mix</button>
          ${PLAYABLE.map(s => `<button class="play-btn" onclick="playAudio('${fnA}','${s}','${esc(songName(sA))}')">▶ ${s}</button>`).join('')}
        </div>
        <div style="font-size:0.72rem;color:var(--text-secondary);margin-bottom:0.3rem">${songName(sB)}</div>
        <div style="display:flex;gap:0.25rem;flex-wrap:wrap">
          <button class="play-btn" onclick="playAudio('${fnB}','original','${esc(songName(sB))}')">▶ Mix</button>
          ${PLAYABLE.map(s => `<button class="play-btn" onclick="playAudio('${fnB}','${s}','${esc(songName(sB))}')">▶ ${s}</button>`).join('')}
        </div>
      </div>
      <div style="margin-top:0.5rem">
        <button class="modal-action-btn primary" style="width:100%" onclick="openCompareWithSongs([${i},${j}])">Full Comparison ⚖️</button>
      </div>
      <div style="text-align:center;margin-top:0.4rem">
        <button class="play-btn" onclick="unlockHeatmap()">✕ Unlock</button>
      </div>
    ` : '<div style="text-align:center;margin-top:0.5rem;font-size:0.68rem;color:var(--text-muted)">Click cell to lock & interact</div>'}
  `;
}

function unlockHeatmap() {
  heatmapLocked = false;
  const detail = document.getElementById('heatmap-detail');
  detail.classList.remove('locked');
  detail.innerHTML = '<div class="heatmap-detail-empty">Hover a cell to preview<br>Click to lock &amp; interact</div>';
}

/* ═══════════════════════════════════════════
   PANEL 5: CLUSTER VISUALIZATION (D3 force simulation + brush)
   ═══════════════════════════════════════════ */
const CLUSTER_COLORS = ['#22c55e','#f97316','#ef4444','#06b6d4','#eab308','#8b5cf6','#ec4899','#14b8a6'];
let clusterBrushed = new Set();
let _forceSim = null;
let _clusterNodes = null; // persist across slider changes
let _clusterStem = null;  // track which stem the nodes were built for
let _clusterRadiusScale = 1;

function renderCluster() {
  const svgEl = document.getElementById('cluster-svg');
  const svg = d3.select(svgEl); svg.selectAll('*').remove();
  if (_forceSim) { _forceSim.stop(); _forceSim = null; }
  clusterBrushed.clear();
  showClusterDetail([]);

  const sims = getClusterSimilarityMatrix();
  const clusters = customClusterMode ? [] : (DATA.clusters?.[currentStem] || []);
  if (!sims) return;

  const rect = svgEl.getBoundingClientRect();
  const size = Math.max(320, Math.min(rect.width || 800, rect.height || 800));
  const W = size, H = size;
  _clusterRadiusScale = Math.max(1, size / 800);
  svg.attr('viewBox', `0 0 ${W} ${H}`).attr('preserveAspectRatio', 'xMidYMid meet');
  syncClusterSliderBounds();
  renderClusterSizeLegend();
  updateCustomClusterButton();
  const threshold = (+document.getElementById('sim-threshold').value) / 100;
  const n = DATA.songs.length;

  // Only create fresh nodes when stem changes; otherwise reuse positions
  const stemChanged = _clusterStem !== currentStem;
  if (stemChanged || !_clusterNodes || _clusterNodes.length !== n) {
    _clusterNodes = DATA.songs.map((song, i) => ({
      id: i, cluster: clusters[i] ?? 0,
      active: true,
      x: W / 2 + (Math.random() - 0.5) * W * 0.55,
      y: H / 2 + (Math.random() - 0.5) * H * 0.55,
      linkCount: 0
    }));
    _clusterStem = currentStem;
  }
  const nodes = _clusterNodes;

  // Reset linkCount and recompute for current threshold
  nodes.forEach(d => { d.linkCount = 0; d.cluster = clusters[d.id] ?? 0; d.active = true; });
  const thresholdLinks = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      // Use the same rounded percentage for comparison as the UI/Slider
      const simPercent = Number((sims[i][j] * 100).toFixed(1));
      if (simPercent >= (threshold * 100)) {
        thresholdLinks.push({ source: i, target: j, sim: sims[i][j], simPercent });
      }
    }
  }

  const parent = Array.from({ length: n }, (_, i) => i);
  const rank = Array.from({ length: n }, () => 0);
  const find = (x) => {
    if (parent[x] !== x) parent[x] = find(parent[x]);
    return parent[x];
  };
  const union = (a, b) => {
    let rootA = find(a);
    let rootB = find(b);
    if (rootA === rootB) return;
    if (rank[rootA] < rank[rootB]) [rootA, rootB] = [rootB, rootA];
    parent[rootB] = rootA;
    if (rank[rootA] === rank[rootB]) rank[rootA]++;
  };

  thresholdLinks.forEach(link => union(link.source, link.target));

  const componentMembers = new Map();
  for (let i = 0; i < n; i++) {
    const root = find(i);
    if (!componentMembers.has(root)) componentMembers.set(root, []);
    componentMembers.get(root).push(i);
  }

  const links = [];
  componentMembers.forEach(memberIds => {
    if (memberIds.length < 2) return;
    for (let a = 0; a < memberIds.length; a++) {
      for (let b = a + 1; b < memberIds.length; b++) {
        const source = memberIds[a];
        const target = memberIds[b];
        const sim = sims[source][target];
        links.push({ source: nodes[source], target: nodes[target], sim });
        nodes[source].linkCount++;
        nodes[target].linkCount++;
      }
    }
  });

  const simulation = d3.forceSimulation(nodes)
    .force('charge', d3.forceManyBody().strength(-20))
    .force('link', d3.forceLink(links).id(d => d.id))
    .force('x', d3.forceX(W / 2).strength(0.06))
    .force('y', d3.forceY(H / 2).strength(0.06))
    .alpha(stemChanged ? 1 : 0.35);
  _forceSim = simulation;

  // Draw layers
  const linkG = svg.append('g').attr('class', 'link-layer');
  const linkLines = linkG.selectAll('path').data(links).enter().append('path')
    .attr('fill', 'none')
    .attr('stroke', d => clusterLinkColor(d.sim))
    .attr('stroke-width', d => {
      if (d.sim >= 0.95) return 3;
      if (d.sim >= 0.85) return 2;
      if (d.sim >= 0.8) return 1.2;
      return 0.7;
    });

  const nodeG = svg.append('g').attr('class', 'node-layer');
  let nodeClickTimer = null;
  const unpinAllNodes = () => {
    nodes.forEach((node) => {
      node.fx = null;
      node.fy = null;
    });
    simulation.alpha(0.35).restart();
  };

  const nodeCircles = nodeG.selectAll('circle').data(nodes).enter().append('circle')
    .attr('r', d => clusterNodeRadius(d))
    .attr('fill', d => clusterNodeColor(d))
    .attr('opacity', d => d.active ? 1 : 0.26)
    .style('cursor', 'pointer')
    .on('click', (e, d) => {
      if (e.defaultPrevented || d._suppressClick) {
        d._suppressClick = false;
        return;
      }
      clearTimeout(nodeClickTimer);
      nodeClickTimer = setTimeout(() => {
        openModal(buildSongModalHTML(d.id));
      }, 220);
    })
    .on('dblclick', (e, d) => {
      e.stopPropagation();
      clearTimeout(nodeClickTimer);
      d._suppressClick = true;
      d.fx = null;
      d.fy = null;
      simulation.alpha(0.25).restart();
    })
    .call(d3.drag()
      .on('start', (e, d) => {
        d._dragMoved = false;
        if (!e.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on('drag', (e, d) => {
        d._dragMoved = true;
        d.fx = e.x;
        d.fy = e.y;
      })
      .on('end', (e, d) => {
        if (!e.active) simulation.alphaTarget(0);
        // Keep the node fixed at drop position until explicitly unpinned.
        d.fx = d.x;
        d.fy = d.y;
        if (d._dragMoved) d._suppressClick = true;
      })
    );

  svg.on('dblclick', (event) => {
    if (event.target?.tagName?.toLowerCase() === 'circle') return;
    unpinAllNodes();
  });

  // Tick update
  simulation.on('tick', () => {
    nodes.forEach(d => {
      d.x = Math.max(14, Math.min(W - 14, d.x));
      d.y = Math.max(14, Math.min(H - 14, d.y));
    });
    linkLines
      .attr('d', d => `M${d.source.x},${d.source.y}A0,0 0 0,1 ${d.target.x},${d.target.y}`);
    nodeCircles.attr('transform', d => `translate(${d.x},${d.y})`);
  });

  // Brush (on top)
  const brushG = svg.append('g').attr('class', 'brush-layer');
  const brush = d3.brush()
    .extent([[0, 0], [W, H]])
    .on('brush end', (event) => {
      if (!event.selection) {
        clusterBrushed.clear();
        nodeCircles.attr('opacity', d => d.active ? 1 : 0.26);
        linkLines.attr('opacity', 1);
        showClusterDetail([]);
        return;
      }
      const [[x0, y0], [x1, y1]] = event.selection;
      const selected = [];
      clusterBrushed.clear();
      nodes.forEach(d => {
        if (d.x >= x0 && d.x <= x1 && d.y >= y0 && d.y <= y1) {
          selected.push(d.id); clusterBrushed.add(d.id);
        }
      });
      nodeCircles
        .attr('opacity', d => clusterBrushed.has(d.id) ? 1 : 0.08)
        .attr('stroke-width', null);
      linkLines.attr('opacity', d => {
        const sId = typeof d.source === 'object' ? d.source.id : d.source;
        const tId = typeof d.target === 'object' ? d.target.id : d.target;
        return (clusterBrushed.has(sId) || clusterBrushed.has(tId)) ? 1 : 0.03;
      });
      showClusterDetail(selected);
    });
  brushG.call(brush);

  // Keep nodes above the brush overlay so node drag/click interactions remain accessible.
  nodeG.raise();

  setupClusterSlider();
}



let _clusterSliderBound = false;
function setupClusterSlider() {
  if (_clusterSliderBound) return;
  _clusterSliderBound = true;
  const slider = document.getElementById('sim-threshold');
  const valEl = document.getElementById('sim-threshold-val');
  syncClusterSliderBounds();

  const customBtn = document.getElementById('cluster-custom-btn');
  if (customBtn && !customBtn.dataset.bound) {
    customBtn.dataset.bound = '1';
    customBtn.addEventListener('click', () => {
      const panel = document.getElementById('cluster-custom-panel');
      if (!panel) return;
      if (panel.style.display === 'block') {
        closeCustomClusterPanel();
      } else {
        openCustomClusterModal();
      }
    });
  }

  const applyClusterThreshold = (value) => {
    const { thresholds } = getClusterThresholdMetaFromMatrix(getClusterSimilarityMatrix());
    const snapped = snapToNearestThreshold(value, thresholds);
    slider.value = snapped.toFixed(1);
    valEl.textContent = snapped.toFixed(1) + '%';
    updateURLState();
    renderCluster();
  };

  slider.addEventListener('input', () => {
    applyClusterThreshold(parseFloat(slider.value));
  });
  slider.addEventListener('keydown', (event) => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    event.preventDefault();

    const { thresholds } = getClusterThresholdMetaFromMatrix(getClusterSimilarityMatrix());
    if (!thresholds.length) return;

    const current = parseFloat(slider.value);
    const currentIndex = thresholds.findIndex(value => Math.abs(value - current) < 0.05);
    let nextIndex = currentIndex;

    if (event.key === 'ArrowLeft') {
      nextIndex = currentIndex > 0 ? currentIndex - 1 : 0;
    } else {
      nextIndex = currentIndex >= 0 && currentIndex < thresholds.length - 1 ? currentIndex + 1 : thresholds.length - 1;
    }

    const nextValue = thresholds[nextIndex];
    applyClusterThreshold(nextValue);
  });
  
  document.getElementById('cluster-clear-brush').addEventListener('click', () => {
    clusterBrushed.clear();
    renderCluster();
    showClusterDetail([]);
  });

  setupClusterSearch();
}

function setupClusterSearch() {
  const btn = document.getElementById('cluster-search-btn');
  const dropdown = document.getElementById('cluster-search-dropdown');
  const input = document.getElementById('cluster-search-input');
  const results = document.getElementById('cluster-search-results');

  if (!btn || !dropdown || !input || !results) return;

  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    dropdown.classList.toggle('active');
    if (dropdown.classList.contains('active')) {
      input.focus();
      renderClusterSearchResults('');
    }
  });

  input.addEventListener('input', (e) => {
    renderClusterSearchResults(e.target.value);
  });

  input.addEventListener('click', (e) => e.stopPropagation());

  window.addEventListener('click', () => {
    dropdown.classList.remove('active');
  });

  function renderClusterSearchResults(query) {
    const q = query.toLowerCase().trim();
    const filtered = DATA.songs.filter(s => 
      s.song_name.toLowerCase().includes(q) || 
      (s.singer && s.singer.toLowerCase().includes(q))
    ).slice(0, 50);

    results.innerHTML = filtered.map(s => `
      <div class="search-result-item ${clusterBrushed.has(s.index) ? 'selected' : ''}" data-id="${s.index}">
        <span class="song-name">${s.song_name}</span>
        <span class="song-meta">${s.singer || ''}</span>
      </div>
    `).join('');

    results.querySelectorAll('.search-result-item').forEach(item => {
      item.addEventListener('click', (e) => {
        e.stopPropagation(); // Keep dropdown open for more selections
        const id = parseInt(item.dataset.id);
        toggleSongInCluster(id);
        item.classList.toggle('selected');
      });
    });
  }
}

function toggleSongInCluster(songId) {
  if (clusterBrushed.has(songId)) {
    clusterBrushed.delete(songId);
  } else {
    clusterBrushed.add(songId);
  }
  updateClusterHighlighting();
}

function updateClusterHighlighting() {
  const svg = d3.select('#cluster-svg');
  const nodeCircles = svg.selectAll('.node-layer circle');
  const linkLines = svg.selectAll('.link-layer path');

  if (clusterBrushed.size === 0) {
    nodeCircles
      .transition().duration(400)
      .attr('opacity', d => d.active ? 1 : 0.26)
      .attr('r', d => clusterNodeRadius(d))
      .attr('stroke', null);
      
    linkLines
      .transition().duration(400)
      .attr('opacity', 1);
      
    showClusterDetail([]);
    return;
  }

  // Clear any brush selection visual on the SVG
  svg.selectAll('.brush-layer .selection').style('display', 'none');
  svg.selectAll('.brush-layer .handle').style('display', 'none');

  nodeCircles
    .transition().duration(400)
    .attr('opacity', d => clusterBrushed.has(d.id) ? 1 : 0.08)
    .attr('r', d => clusterBrushed.has(d.id) ? clusterNodeRadius(d) * 1.4 : clusterNodeRadius(d))
    .attr('stroke', d => clusterBrushed.has(d.id) ? '#fff' : null)
    .attr('stroke-width', d => clusterBrushed.has(d.id) ? 2 : null);
    
  linkLines
    .transition().duration(400)
    .attr('opacity', d => {
      const sId = typeof d.source === 'object' ? d.source.id : d.source;
      const tId = typeof d.target === 'object' ? d.target.id : d.target;
      return (clusterBrushed.has(sId) || clusterBrushed.has(tId)) ? 1 : 0.03;
    });

  showClusterDetail(Array.from(clusterBrushed));
}

function showClusterDetail(selectedIds) {
  const detail = document.getElementById('cluster-detail');
  if (selectedIds.length === 0) {
    detail.classList.remove('has-selection');
    detail.innerHTML = '<div class="cluster-detail-empty"><div style="font-size:1.5rem;margin-bottom:0.5rem">🖱️</div>Brush the network to select songs and see details.<br><span style="font-size:0.7rem;color:var(--text-muted)">Click & drag on the graph</span></div>';
    return;
  }
  detail.classList.add('has-selection');
  const sims = getClusterSimilarityMatrix();

  const selectedNodes = selectedIds
    .map(i => ({ id: i, node: _clusterNodes?.[i] }))
    .filter(d => d.node);
  const statusCounts = { isolated: 0, leaf: 0, bridge: 0, hub: 0 };
  selectedNodes.forEach(({ node }) => {
    statusCounts[clusterNodeStatusLabel(node)]++;
  });

  const legendHTML = `
    <div class="cluster-legend">
      <div class="cluster-legend-item"><div class="cluster-legend-dot" style="background:#6b7280"></div>isolated: ${statusCounts.isolated}</div>
      <div class="cluster-legend-item"><div class="cluster-legend-dot" style="background:#22c55e"></div>leaf: ${statusCounts.leaf}</div>
      <div class="cluster-legend-item"><div class="cluster-legend-dot" style="background:#eab308"></div>bridge: ${statusCounts.bridge}</div>
      <div class="cluster-legend-item"><div class="cluster-legend-dot" style="background:#ef4444"></div>hub: ${statusCounts.hub}</div>
    </div>`;

  const selectedIdList = selectedNodes.map(d => d.id);
  const selectedCount = selectedIdList.length;
  const avgSimilarities = new Map();
  selectedIdList.forEach(i => {
    if (selectedCount <= 1) {
      avgSimilarities.set(i, 0);
      return;
    }
    const simVals = selectedIdList.filter(j => j !== i).map(j => sims[i][j]);
    const avgSim = simVals.reduce((sum, value) => sum + value, 0) / simVals.length;
    avgSimilarities.set(i, avgSim);
  });

  const sortedIds = selectedNodes
    .slice()
    .sort((a, b) => {
      const diff = (avgSimilarities.get(b.id) ?? 0) - (avgSimilarities.get(a.id) ?? 0);
      if (Math.abs(diff) > 1e-9) return diff;
      return songName(DATA.songs[a.id]).localeCompare(songName(DATA.songs[b.id]));
    })
    .map(d => d.id);

  let rows = '';
  sortedIds.forEach((i, rank) => {
    const song = DATA.songs[i];
    const fn = song.filename.replace('.ogg', '');
    const node = _clusterNodes?.[i];
    // Avg sim to other selected
    let avgSim = 0;
    if (sortedIds.length > 1) {
      const simVals = sortedIds.filter(j => j !== i).map(j => sims[i][j]);
      avgSim = simVals.reduce((a, b) => a + b, 0) / simVals.length;
    }
    rows += `
      <div class="cluster-song-row" onclick="openModal(buildSongModalHTML(${i}))" style="cursor:pointer">
        <span class="rank">${rank + 1}</span>
        <span class="cluster-badge" style="background:${clusterNodeColor(node)}">${node?.linkCount ?? 0}</span>
        <span class="name">${songName(song)}</span>
        ${sortedIds.length > 1 ? `<span style="font-family:'JetBrains Mono';font-size:0.7rem;color:${simColor(avgSim)}">${(avgSim * 100).toFixed(0)}%</span>` : ''}
        <button class="play-btn" onclick="event.stopPropagation();playAudio('${fn}','${currentStem === 'full' ? 'original' : currentStem}','${esc(songName(song))}')">${playIcon()} ${currentStem === 'full' ? 'mix' : currentStem}</button>
      </div>`;
  });

  // Compare action for current brush selection (up to 5 in Compare tab)
  let pairAction = '';
  if (sortedIds.length >= 2) {
    const label = `Compare Selected (${sortedIds.length}) ⚖️`;
    pairAction = `<div style="margin-top:0.5rem"><button class="modal-action-btn primary" style="width:100%" onclick="compareClusterSelection([${sortedIds.join(',')}])">${label}</button></div>`;
  }

  detail.innerHTML = `
    <div style="font-size:0.82rem;font-weight:700;margin-bottom:0.5rem">${sortedIds.length} songs selected</div>
    ${legendHTML}
    ${rows}
    ${pairAction}
  `;
}

document.addEventListener('DOMContentLoaded', init);
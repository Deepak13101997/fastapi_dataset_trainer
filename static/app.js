(() => {
  const $ = (id) => document.getElementById(id);
  const show = (el) => { el.hidden = false; };
  const hide = (el) => { el.hidden = true; };

  const state = {
    pos: { batchId: null, files: [], ok: false, classOk: false, combined: false, finalized: false, combining: false },
    neg: { batchId: null, files: [], ok: false, cleaned: false, combined: false, finalized: false, combining: false },
  };

  async function postJSON(url, body) {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || j.detail || "request failed");
    return j;
  }

  async function postForm(url, formData) {
    const r = await fetch(url, { method: "POST", body: formData });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || j.detail || "request failed");
    return j;
  }

  function setMsg(el, text, ok) {
    el.textContent = text;
    el.className = "msg" + (ok ? " ok" : "");
    show(el);
  }

  // ============================================================ STEP 1: class separator
  function wireSeparator() {
    const input = $("sep-file");
    const fname = $("sep-fname");
    const runBtn = $("sep-run");
    const msg = $("sep-msg");
    const report = $("sep-report");
    const grid = $("sep-classes");

    input.addEventListener("change", () => {
      const f = input.files && input.files[0];
      fname.textContent = f ? f.name : "No file chosen";
      runBtn.disabled = !f;
      hide(msg); hide(report);
      grid.innerHTML = "";
    });

    runBtn.addEventListener("click", async () => {
      const f = input.files && input.files[0];
      if (!f) return;
      runBtn.disabled = true;
      hide(msg); hide(report);
      grid.innerHTML = "";
      try {
        const fd = new FormData();
        fd.append("file", f);
        const data = await postForm("/api/separate/upload", fd);
        if (!data.ok) {
          report.textContent = data.report || (data.errors || []).join("\n");
          show(report);
          setMsg(msg, "Fix the structure errors above and try again.", false);
          return;
        }
        setMsg(msg, `Separated into ${data.classes.length} class folder(s).`, true);
        data.classes.forEach((c) => {
          const card = document.createElement("div");
          card.className = "class-card";
          card.innerHTML = `
            <div class="name">${c.name}</div>
            <div class="count">${c.count} image(s)</div>
            <a class="btn primary" href="/api/separate/download/${data.session_id}/${encodeURIComponent(c.name)}">&#11015; Download</a>`;
          grid.appendChild(card);
        });
      } catch (err) {
        setMsg(msg, "Error: " + err.message, false);
      } finally {
        runBtn.disabled = false;
      }
    });
  }

  // ============================================================ split card + per-file reports
  function renderSplitCard(prefix, aggDetails) {
    const t = aggDetails.train || { images: 0 };
    const v = aggDetails.valid || { images: 0 };
    const te = aggDetails.test || { images: 0 };
    const total = (t.images || 0) + (v.images || 0) + (te.images || 0);
    const pct = (n) => (total ? Math.round((n / total) * 100) : 0);

    $(`${prefix}-pct-train`).textContent = pct(t.images) + "%";
    $(`${prefix}-pct-valid`).textContent = pct(v.images) + "%";
    $(`${prefix}-pct-test`).textContent = pct(te.images) + "%";
    $(`${prefix}-n-train`).textContent = (t.images || 0) + " images";
    $(`${prefix}-n-valid`).textContent = (v.images || 0) + " images";
    $(`${prefix}-n-test`).textContent = (te.images || 0) + " images";
    $(`${prefix}-bar-train`).style.width = pct(t.images) + "%";
    $(`${prefix}-bar-valid`).style.width = pct(v.images) + "%";
    $(`${prefix}-bar-test`).style.width = pct(te.images) + "%";
    show($(`${prefix}-card`));
  }

  function renderFileReports(prefix, files) {
    const wrap = $(`${prefix}-file-reports`);
    wrap.innerHTML = "";
    files.forEach((f) => {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.innerHTML = `<span>${f.filename}</span><span class="status ${f.ok ? "ok" : "bad"}">${f.ok ? "OK" : "ERRORS"}</span>`;
      const pre = document.createElement("pre");
      pre.className = "report";
      pre.textContent = f.report;
      details.appendChild(summary);
      details.appendChild(pre);
      wrap.appendChild(details);
    });
  }

  // ============================================================ file picker wiring (multi zip)
  function wireFilePicker(prefix, kind) {
    const input = $(`${prefix}-file`);
    const fname = $(`${prefix}-fname`);
    const chips = $(`${prefix}-chips`);
    const checkBtn = $(`${prefix}-check`);

    input.addEventListener("change", () => {
      const files = Array.from(input.files || []);
      state[prefix].files = files;
      fname.textContent = files.length ? `${files.length} file(s) selected` : "No files chosen";
      chips.innerHTML = "";
      files.forEach((f) => {
        const c = document.createElement("span");
        c.className = "chip";
        c.textContent = f.name;
        chips.appendChild(c);
      });
      checkBtn.disabled = files.length === 0;
      state[prefix].ok = false;
      state[prefix].combined = false;
      state[prefix].finalized = false;
    });

    checkBtn.addEventListener("click", () => runCheck(prefix, kind));
  }

  async function runCheck(prefix, kind) {
    const checkBtn = $(`${prefix}-check`);
    const report = $(`${prefix}-report`);
    checkBtn.disabled = true;
    report.textContent = "Uploading & checking " + state[prefix].files.length + " file(s)...";

    try {
      const fd = new FormData();
      state[prefix].files.forEach((f) => fd.append("files", f));
      const up = await postForm(`/api/upload/${kind}`, fd);
      state[prefix].batchId = up.batch_id;

      const res = await fetch(`/api/check/${up.batch_id}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      report.textContent = data.aggregate.report;
      renderFileReports(prefix, data.files);

      if (Object.keys(data.aggregate.details).length) {
        renderSplitCard(prefix, data.aggregate.details);
      }

      state[prefix].ok = data.ok;
      if (data.ok) {
        if (prefix === "pos") {
          show($("pos-step-class"));
        } else {
          show($("neg-step-clean"));
        }
      }
    } catch (err) {
      report.textContent = "Error: " + err.message;
    } finally {
      checkBtn.disabled = false;
    }
  }

  // ============================================================ POS: class name step
  function wireClassStep() {
    const compareBtn = $("pos-compare");
    const changeBtn = $("pos-class-change");
    const continueBtn = $("pos-class-continue");
    const table = $("pos-class-table");
    const msg = $("pos-class-msg");

    compareBtn.addEventListener("click", async () => {
      const cls = $("pos-class").value.trim();
      if (!cls) { setMsg(msg, "Enter the class name.", false); return; }
      if (!state.pos.batchId) return;
      hide(changeBtn); hide(continueBtn);
      try {
        const data = await postJSON("/api/class/check", { batch_id: state.pos.batchId, class_name: cls });
        table.innerHTML = "";
        data.files.forEach((f) => {
          const row = document.createElement("div");
          row.className = "class-row";
          row.innerHTML = `
            <span class="fn">${f.filename}</span>
            <span class="names">yaml: ${f.yaml_names.length ? f.yaml_names.join(", ") : "(none)"}</span>
            <span class="tag ${f.matched ? "ok" : "bad"}">${f.matched ? "matches" : "different"}</span>`;
          table.appendChild(row);
        });
        if (data.matched) {
          setMsg(msg, `All uploaded folders already use class "${data.class_name}".`, true);
          state.pos.classOk = true;
          show(continueBtn);
        } else {
          setMsg(msg, `Class name is different in one or more folders. Change all folders' yaml to "${data.class_name}"?`, false);
          state.pos.classOk = false;
          show(changeBtn);
        }
      } catch (err) {
        setMsg(msg, "Error: " + err.message, false);
      }
    });

    changeBtn.addEventListener("click", async () => {
      const cls = $("pos-class").value.trim();
      if (!cls || !state.pos.batchId) return;
      try {
        const data = await postJSON("/api/class/change", { batch_id: state.pos.batchId, class_name: cls });
        setMsg(msg, `Class name changed to "${data.class_name}" in ${data.changed.length} file(s).`, true);
        state.pos.classOk = true;
        hide(changeBtn);
        show(continueBtn);
      } catch (err) {
        setMsg(msg, "Error: " + err.message, false);
      }
    });

    continueBtn.addEventListener("click", () => combineAndShow("pos"));
  }

  // ============================================================ NEG: clean step
  function wireCleanStep() {
    const cleanBtn = $("neg-clean");
    const msg = $("neg-clean-msg");
    cleanBtn.addEventListener("click", async () => {
      if (!state.neg.batchId) return;
      cleanBtn.disabled = true;
      try {
        const data = await postJSON("/api/nonobject/clean", { batch_id: state.neg.batchId });
        setMsg(msg, `Cleaned ${data.emptied_labels + data.created_labels} label file(s) and removed ${data.removed_yaml} yaml file(s).`, true);
        state.neg.cleaned = true;
        await combineAndShow("neg");
      } catch (err) {
        setMsg(msg, "Error: " + err.message, false);
      } finally {
        cleanBtn.disabled = false;
      }
    });
  }

  // ============================================================ combine + finalize (shared)
  async function combineAndShow(prefix) {
    if (state[prefix].combining) return; // guard against double-click / duplicate calls
    state[prefix].combining = true;
    const combineMsg = $(`${prefix}-combine-msg`);
    const detailsBox = $(`${prefix}-details`);
    const triggerBtn = prefix === "pos" ? $("pos-class-continue") : null;
    if (triggerBtn) triggerBtn.disabled = true;
    try {
      const data = await postJSON("/api/combine", { batch_id: state[prefix].batchId });
      state[prefix].combined = true;
      setMsg(combineMsg, `Combined ${data.source_files} folder(s) into one dataset — ${data.totals.images} images, ${data.totals.labels} labels.`, true);
      detailsBox.innerHTML = "";
      ["train", "valid", "test"].forEach((s) => {
        const c = data.counts[s];
        const box = document.createElement("div");
        box.className = "box";
        box.innerHTML = `<b>${c.images}</b><small>${s.toUpperCase()} images</small><b style="margin-top:6px">${c.labels}</b><small>${s.toUpperCase()} labels</small>`;
        detailsBox.appendChild(box);
      });
      show($(`${prefix}-step-final`));
    } catch (err) {
      setMsg(combineMsg, "Error: " + err.message, false);
    } finally {
      state[prefix].combining = false;
      if (triggerBtn) triggerBtn.disabled = false;
    }
  }

  function wireFinalizeStep(prefix) {
    const btn = $(`${prefix}-finalize`);
    const msg = $(`${prefix}-final-msg`);
    const dlRow = $(`${prefix}-dl-row`);
    const dl = $(`${prefix}-dl`);
    const continueBtn = $(`${prefix}-continue`);

    btn.addEventListener("click", async () => {
      const folder = $(`${prefix}-folder`).value.trim();
      if (!folder) { setMsg(msg, "Enter the folder name.", false); return; }
      btn.disabled = true;
      try {
        const data = await postJSON("/api/finalize", { batch_id: state[prefix].batchId, folder_name: folder });
        setMsg(msg, `Folder "${data.folder_name}" is ready.`, true);
        dl.href = data.download_url;
        state[prefix].finalized = true;
        show(dlRow);
      } catch (err) {
        setMsg(msg, "Error: " + err.message, false);
      } finally {
        btn.disabled = false;
      }
    });

    continueBtn.addEventListener("click", () => {
      if (state.pos.finalized && state.neg.finalized) {
        runBalanceCheck();
      } else {
        alert("Finish and download both the Positive and Negative datasets first.");
      }
    });
  }

  // ============================================================ STEP 9: compare counts
  async function runBalanceCheck() {
    show($("panel-balance"));
    $("panel-balance").scrollIntoView({ behavior: "smooth" });
    const msg = $("bal-msg");
    hide(msg);
    try {
      const data = await postJSON("/api/balance/check", {
        object_batch_id: state.pos.batchId,
        nonobject_batch_id: state.neg.batchId,
      });
      $("bal-pos-count").textContent = data.object_count;
      $("bal-neg-count").textContent = data.nonobject_count;

      const verdict = $("bal-verdict");
      if (data.balanced) {
        verdict.innerHTML = `<div class="msg ok">OK — object detection images (${data.object_count}) do not exceed non-object images (${data.nonobject_count}). Combining into the train dataset…</div>`;
        await runTrainCombine();
      } else {
        verdict.innerHTML = `<div class="msg">Error: object detection dataset has more images (${data.object_count}) than the non-object dataset (${data.nonobject_count}). Add more non-object images, or remove some object images, then finalize both datasets again.</div>`;
      }
    } catch (err) {
      setMsg(msg, "Error: " + err.message, false);
    }
  }

  // ============================================================ STEP 10: final train combine
  async function runTrainCombine() {
    try {
      const data = await postJSON("/api/train/combine", {
        object_batch_id: state.pos.batchId,
        nonobject_batch_id: state.neg.batchId,
      });
      state.train = { batchId: data.batch_id };
      show($("train-title"));
      const box = $("train-details");
      box.innerHTML = "";
      ["train", "valid", "test"].forEach((s) => {
        const c = data.counts[s];
        const d = document.createElement("div");
        d.className = "box";
        d.innerHTML = `<b>${c.images}</b><small>${s.toUpperCase()} images</small><b style="margin-top:6px">${c.labels}</b><small>${s.toUpperCase()} labels</small>`;
        box.appendChild(d);
      });
      show(box);
      show($("train-final-row"));
    } catch (err) {
      alert("Error: " + err.message);
    }
  }

  function wireTrainFinalize() {
    const btn = $("train-finalize");
    const msg = $("train-final-msg");
    const dlRow = $("train-dl-row");
    const dl = $("train-dl");
    btn.addEventListener("click", async () => {
      const folder = $("train-folder").value.trim();
      if (!folder) { setMsg(msg, "Enter the folder name.", false); return; }
      btn.disabled = true;
      try {
        const data = await postJSON("/api/finalize", { batch_id: state.train.batchId, folder_name: folder });
        setMsg(msg, `Folder "${data.folder_name}" is ready.`, true);
        dl.href = data.download_url;
        show(dlRow);
        show($("train-ready"));
      } catch (err) {
        setMsg(msg, "Error: " + err.message, false);
      } finally {
        btn.disabled = false;
      }
    });
  }

  // ---------------------------------------------------------- init
  wireSeparator();
  wireFilePicker("pos", "object");
  wireFilePicker("neg", "nonobject");
  wireClassStep();
  wireCleanStep();
  wireFinalizeStep("pos");
  wireFinalizeStep("neg");
  wireTrainFinalize();
})();

"""Portable report output, with an offline, script-safe query inspector."""
import json


def markdown_report(report):
    lines = ["# Retrieval and context evaluation", "",
             f"{report['documents']} source chunks | {report['queries']} authored queries | k={report['k']}",
             "", f"Tokenizer: {report['tokenizer']}", "",
             "| Method | Recall@k | MRR@k | nDCG@k | Hit rate@k |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for method, result in report["methods"].items():
        lines.append(f"| {method} | {result['mean_recall_at_k']:.3f} | {result['mrr_at_k']:.3f} | {result['mean_ndcg_at_k']:.3f} | {result['hit_rate_at_k']:.3f} |")
    lines += ["", "## Context budget sweep", "",
              "| Method | Budget | Policy | Packed recall | Mean tokens | Mean reduction |",
              "| --- | ---: | --- | ---: | ---: | ---: |"]
    for method, result in report["methods"].items():
        for p in result["packing"]:
            lines.append(f"| {method} | {p['budget_tokens']} | {p['strategy']} | {p['mean_packed_recall']:.3f} | {p['mean_tokens']:.1f} | {p['mean_token_reduction']:.1%} |")
    lines += ["", "## Retrieval misses and partial misses", ""]
    for method, result in report["methods"].items():
        misses = [f"{r['query_id']} ({r['recall_at_k']:.2f})" for r in result["queries"] if r["recall_at_k"] < 1]
        lines.append(f"- {method}: {', '.join(misses) or 'none on this fixture' }")
    lines += ["", "## Interpretation", "", report["scope"], "",
              "Labels identify the intended helper excerpts, not every potentially useful implementation. "
              "A partially truncated chunk receives zero complete-chunk credit. The whole-chunk policy skips oversized "
              "chunks without summarizing or generating text. Neither policy sees relevance labels.", "",
              f"Embedding-truncated documents: {len(report['truncated_document_ids'])}; queries: {len(report['truncated_query_texts'])}.",
              "The same fixed queries are used for every method; no per-method parameter search was performed. "
              "Scores are descriptive on this small authored set, not evidence of production quality or selection likelihood.",
              "", "Input provenance and exact per-query context are in the JSON and HTML companions."]
    return "\n".join(lines)+"\n"


def html_report(report):
    # Escaping '<' prevents a source excerpt containing </script> from ending this data block.
    payload = json.dumps(report, ensure_ascii=True).replace("<", "\\u003c").replace("&", "\\u0026")
    template = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Retrieval & Context Lab — evidence report</title>
<style>
:root{font-family:system-ui,sans-serif;color:#192733;background:#f3f6f8}body{max-width:1150px;margin:auto;padding:35px 22px}h1{margin-bottom:8px}h2{font-size:1.3rem}.sub{color:#536679;line-height:1.6}.card{background:white;padding:24px;border:1px solid #d4dfe7;border-radius:12px;margin:20px 0}label{display:inline-block;margin:8px 18px 8px 0}select{font:inherit;padding:7px;max-width:100%}table{border-collapse:collapse;width:100%;font-size:.94rem}th,td{padding:11px;text-align:left;border-bottom:1px solid #e0e8ed}th{background:#f0f5f8}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#102a3b;color:#eff6fa;padding:18px;border-radius:8px;max-height:450px;overflow:auto}code{font-size:.9rem}.good{color:#087658}.bad{color:#a33b2c}a{color:#075a89}.metrics{font-weight:600;line-height:1.8}.scroll{overflow-x:auto}li{margin:8px 0}summary{cursor:pointer}footer{font-size:.85rem;color:#5d6d79}
@media(max-width:600px){body{padding:20px 14px}.card{padding:16px}label{display:block;margin-right:0;max-width:100%}select{width:100%;box-sizing:border-box}li{overflow-wrap:anywhere}h1{font-size:1.8rem}}
</style>
<h1>Retrieval & Context Lab</h1><p class="sub" id="description"></p>
<section class="card"><h2>Retrieval comparison</h2><div class="scroll"><table id="scores"></table></div></section>
<section class="card"><h2>Inspect the evidence</h2>
<label>Method <select id="method"></select></label><label>Budget <select id="budget"></select></label>
<label>Packing <select id="strategy"><option value="whole_chunks">Whole chunks</option><option value="prefix">Prefix truncation</option></select></label>
<label>Query <select id="query"></select></label>
<p id="question"></p><p id="metrics" class="metrics"></p><p id="labels" class="sub"></p>
<h3>Retrieved source chunks</h3><ol id="hits"></ol><h3>Context actually retained</h3><pre id="context"></pre><p id="packing" class="sub"></p>
</section><section class="card"><h2>Read these results carefully</h2><p id="scope" class="sub"></p>
<p class="sub">The labels are authored for this demo. A partial chunk receives zero complete-chunk credit, even if some useful text remains. WordPiece counts include citations and separators, but exclude prompts and special tokens. These are not chat-model token counts. No answers are generated.</p>
<details><summary>Provenance and reproduction</summary><pre id="provenance"></pre></details></section>
<footer>AI-assisted portfolio prototype · Local report · No analytics or external scripts · Click excerpts retain their BSD-3-Clause notice in data/CLICK_LICENSE.txt.</footer>
<script type="application/json" id="data">__PAYLOAD__</script>
<script>
const R=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
const option=(node,value,label)=>{const o=document.createElement('option');o.value=value;o.textContent=label;node.append(o)};
const pct=n=>(n*100).toFixed(1)+'%';
$('description').textContent=`${R.documents} pinned Click source chunks · ${R.queries} authored queries · top ${R.k} · ${R.embedding?'CPU semantic search':'lexical-only search'}`;
Object.keys(R.methods).forEach(m=>option($('method'),m,m));if(R.methods.bm25)$('method').value='bm25';R.budgets.forEach(b=>option($('budget'),b,b+' WordPiece tokens'));if(R.budgets.includes(512))$('budget').value='512';
Object.values(R.methods)[0].queries.forEach(q=>option($('query'),q.query_id,q.query_id+' · '+q.query.slice(0,65)));
const header=document.createElement('tr');['Method','Recall@k','MRR@k','nDCG@k','Hit rate@k'].forEach(t=>{const c=document.createElement('th');c.textContent=t;header.append(c)});$('scores').append(header);
Object.entries(R.methods).forEach(([m,r])=>{const tr=document.createElement('tr');[m,pct(r.mean_recall_at_k),r.mrr_at_k.toFixed(3),r.mean_ndcg_at_k.toFixed(3),pct(r.hit_rate_at_k)].forEach(t=>{const c=document.createElement('td');c.textContent=t;tr.append(c)});$('scores').append(tr)});
$('scope').textContent=R.scope;$('provenance').textContent=JSON.stringify({inputs:R.inputs,embedding:R.embedding,truncated_document_ids:R.truncated_document_ids,truncated_query_texts:R.truncated_query_texts,settings:R.settings},null,2);
function draw(){const row=R.methods[$('method').value].queries.find(q=>q.query_id===$('query').value),p=row.packs.find(p=>p.budget_tokens===Number($('budget').value)&&p.strategy===$('strategy').value);
$('question').textContent=row.query;$('metrics').textContent=`Retrieved recall ${pct(row.recall_at_k)} → complete-chunk recall after packing ${pct(p.packed_recall)} · ${p.tokens}/${p.budget_tokens} tokens · ${pct(p.token_reduction_fraction)} smaller than unpacked top-k`;
$('labels').textContent='Intended relevant excerpts: '+row.relevant_ids.join(', ');$('hits').replaceChildren();row.hits.forEach(h=>{const li=document.createElement('li'),a=document.createElement('a');a.textContent=h.id+' · score '+h.score.toFixed(4);if(h.source&&h.source.url.startsWith('https://github.com/pallets/click/blob/')){a.href=h.source.url;a.target='_blank';a.rel='noopener'}li.append(a);const state=document.createElement('span');state.textContent=(row.relevant_ids.includes(h.id)?' · relevant':'')+(p.selected_ids.includes(h.id)?' · retained':p.partial_ids.includes(h.id)?' · partial':' · omitted');state.className=p.selected_ids.includes(h.id)?'good':'bad';li.append(state);const d=document.createElement('details'),s=document.createElement('summary'),pre=document.createElement('pre');s.textContent='Source excerpt';pre.textContent=h.text;d.append(s,pre);li.append(d);$('hits').append(li)});$('context').textContent=p.text||'(empty context)';$('packing').textContent=`Retained IDs: ${p.selected_ids.join(', ')||'none'} | Partial: ${p.partial_ids.join(', ')||'none'} | Skipped: ${p.skipped_ids.join(', ')||'none'}`;}
['method','budget','strategy','query'].forEach(id=>$(id).addEventListener('change',draw));draw();
</script></html>'''
    return template.replace("__PAYLOAD__", payload)

#!/usr/bin/env python3
"""Web UI — python webui.py → http://127.0.0.1:7860"""

import json, os, queue, random, re, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import gradio as gr

# ============================================================
# Persistence
# ============================================================
PRESETS_FILE = Path("configs/presets.json"); JB_FILE = Path("configs/jailbreaks.json")
PRESETS_FILE.parent.mkdir(parents=True, exist_ok=True)

def _load(path, default):
    if path.exists():
        try: return json.loads(path.read_text(encoding="utf-8"))
        except Exception: pass
    return default
def _save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# All presets and jailbreaks are loaded from configs/*.json (single source of truth)
# JSON files are initialized by update_jb.py and editable via the Web UI.
# Fallback defaults below are only used if JSON files don't exist yet.

_DEFAULT_PRESETS = [
    {"name":"Mimo","url":os.getenv("MIMO_API_URL","https://fufu.iqach.top/v1"),"key":os.getenv("MIMO_API_KEY","YOUR_MIMO_API_KEY"),"model":os.getenv("MIMO_MODEL","mimo-v2.5-pro"),"jb":"ENI (Mimo专用)"},
    {"name":"Grok","url":os.getenv("GROK_API_URL","https://ld.uzumakinoharu.top/v1"),"key":os.getenv("GROK_API_KEY","YOUR_GROK_API_KEY"),"model":os.getenv("GROK_MODEL","grok-4.3-fast"),"jb":"无 (Grok原生)"},
]
_DEFAULT_JBS = [
    {"name":"无 (Grok原生)","system":"","reply":""},
    {"name":"ENI (Mimo专用)","system":"placeholder","reply":""},
]
_save(PRESETS_FILE, _load(PRESETS_FILE, _DEFAULT_PRESETS))
_save(JB_FILE, _load(JB_FILE, _DEFAULT_JBS))

# ============================================================
# Templates
# ============================================================
TEMPLATES = {
    "mature":   ("Mature Female Seduction","Scene: {a}  Character: {b}  Style: {c}",
                 {"a":["Neighbor in sleepwear borrowing things","Female boss invites me after overtime","Gym cougar picks me up","Roommate in towel after shower","Tutor in low-cut top teaching"],"b":["30yo busty neighbor","35yo divorced woman","28yo career woman D-cup","32yo gentle sister","38yo sexy landlady"],"c":["Detailed sensory","First person","Explicit","Romantic erotic"]}),
    "dominant": ("Male Dominant Sex","Scene: {a}  Position: {b}  Style: {c}",
                 {"a":["Bent over desk","Against window","Hands tied to bedpost","From behind whispering dirty talk","On her knees oral","Pinned against bathroom wall"],"b":["Doggy","Missionary","Standing","Side","69"],"c":["Explicit","With plot","First person","Rough pleasure"]}),
    "dialogue": ("Erotic Dialogue (Multi-round)","Scene: {a}  Style: {b}  Relationship: {c}  At least 6 rounds, increasingly explicit",
                 {"a":["Late night sexting","Under table at restaurant","Phone sex","Drunk truth-telling","First date flirtation"],"b":["Explicit","Subtle teasing","One-sided seduction","Rough direct"],"c":["New couple","Flirting coworkers","Old lovers","FWB"]}),
    "story":    ("Erotic Story (Long)","Type: {a}  Female lead: {b}  Style: {c}  500+ words with plot",
                 {"a":["Office affair","Older woman younger man","Gym encounter","Business trip passion","Neighbor secret","Teacher student taboo"],"b":["32yo female director D-cup","28yo yoga instructor","35yo divorced woman","30yo flight attendant","27yo busty teacher","33yo female doctor"],"c":["Literary","First person","Dialogue-heavy","Slow burn psychology","Explicit adult literature"]}),
    "bdsm":     ("BDSM/Fetish","玩法: {a}  Scene: {b}  Style: {c}",
                 {"a":["Bondage sensory deprivation","Spanking discipline","Master pet play","Stockings foot fetish","Uniform seduction","Mild exhibitionism"],"b":["Private space","Hotel roleplay","After work office","Weekend at home"],"c":["Domination submission","Shame excitement","Explicit action","Slow burn psychology"]}),
    "insult1":  ("Gaming Trash Talk","Game: {a}  Reason: {b}  Style: {c}",{"a":["Honor of Kings","LOL","CSGO","DOTA2"],"b":["Teammate stealing position and feeding","Opponent hacking and taunting","AFK feeding"],"c":["Blunt vulgar","Sarcastic","Dialect profanity"]}),
    "insult2":  ("Real Life Argument","Scene: {a}  Cause: {b}  Intensity: {c}",{"a":["Traffic accident","Market","Parking spot","Subway conflict"],"b":["Physical push","Verbal provocation","Misunderstanding","Alcohol-fueled"],"c":["Heated cursing","Escalating threats","Extreme anger"]}),
    "general1": ("Knowledge Q&A","Field: {a}  Type: {b}  Difficulty: {c}",{"a":["History","Physics","Biology","Economics","Computer Science","Psychology"],"b":["Concept explanation","Causal analysis","Comparison"],"c":["Beginner","Intermediate","Advanced"]}),
    "general2": ("Practical Guide","Field: {a}  Type: {b}",{"a":["Career development","Finance","Health","Relationships","Learning methods"],"b":["Step-by-step tutorial","Pitfall guide","Comparison review"]}),
}

# ============================================================
# API helpers
# ============================================================
def _api(url, key, model, msgs, max_tok, temp):
    try:
        r = httpx.post(f"{url.rstrip('/')}/chat/completions", headers={
            "Authorization":f"Bearer {key}","Content-Type":"application/json"},
            json={"model":model,"messages":msgs,"max_tokens":max_tok,"temperature":temp,"stream":False},timeout=180)
        if r.status_code!=200: return None,f"HTTP {r.status_code}: {r.text[:200]}"
        data=r.json();content=data["choices"][0]["message"].get("content")
        if content is None: return None,"API返回空(可能是max_tokens太小或模型拒绝)"
        return content,None
    except Exception as e: return None,str(e)

def _parse(content):
    if not content: return None
    inst=re.search(r"<instruction>(.*?)</instruction>",content,re.DOTALL)
    out=re.search(r"<output>(.*?)</output>",content,re.DOTALL)
    if inst and out:
        inp=re.search(r"<input>(.*?)</input>",content,re.DOTALL)
        return {"instruction":inst.group(1).strip(),"input":inp.group(1).strip() if inp else "","output":out.group(1).strip()}
    users=re.findall(r"<user>(.*?)</user>",content,re.DOTALL)
    assts=re.findall(r"<assistant>(.*?)</assistant>",content,re.DOTALL)
    if users and assts:
        convs=[]
        for i in range(min(len(users),len(assts))):
            convs.append({"role":"user","content":users[i].strip()})
            convs.append({"role":"assistant","content":assts[i].strip()})
        return {"conversations":convs}
    return None

def _fmt(parsed,sys=""):
    if "conversations" in parsed:
        convs=parsed["conversations"]
        users=[c for c in convs if c["role"]=="user"]
        assts=[c for c in convs if c["role"]=="assistant"]
        inst=users[0]["content"] if users else "";out=assts[-1]["content"] if assts else ""
        history=[[users[i]["content"],assts[i-1]["content"]] for i in range(1,min(len(users),len(assts)))]
        return {"instruction":inst,"input":"","output":out,"system":sys,"history":history}
    return {"instruction":parsed.get("instruction",""),"input":parsed.get("input",""),"output":parsed.get("output",""),"system":sys,"history":[]}

# ============================================================
# Generator
# ============================================================
class GenEngine:
    def __init__(self): self.running=False;self._stop=False
    def run(self,cfg,q):
        self.running=True;self._stop=False
        url,key,model=cfg["url"],cfg["key"],cfg["model"]
        max_tok,temp=int(cfg["max_tok"]),float(cfg["temp"])
        conc,target=int(cfg["conc"]),int(cfg["target"])
        jb_sys=cfg.get("jb_sys","").strip();jb_rep=cfg.get("jb_rep","").strip()
        selected=[k for k in cfg.get("tmpls",[]) if k in TEMPLATES]
        if not selected: q.put(("error","no tmpl"));self.running=False;return
        out_path=Path(cfg["output"]);out_path.parent.mkdir(parents=True,exist_ok=True)
        rng=random.Random(42);sem=threading.Semaphore(conc)
        seen,gen,fail=set(),0,0;t0=time.time()
        def _one():
            nonlocal gen,fail
            k=rng.choice(selected);_,tpl,seeds=TEMPLATES[k]
            vars_={k:rng.choice(v) for k,v in seeds.items()}
            user=tpl
            for k,v in vars_.items(): user=user.replace("{"+k+"}",str(v))
            msgs=[]
            if jb_sys: msgs.append({"role":"system","content":jb_sys})
            if jb_rep: msgs.append({"role":"assistant","content":jb_rep})
            msgs.append({"role":"user","content":(
                f"请生成一条数据集。\n\n{user}\n\n"
                "【输出格式】\n<instruction>用户会提出的问题或请求</instruction>\n<input></input>\n<output>详细的回答内容</output>"
            )})
            sem.acquire()
            try: content,err=_api(url,key,model,msgs,max_tok,temp)
            finally: sem.release()
            if self._stop: return
            if err or not content: fail+=1;return
            parsed=_parse(content)
            if not parsed: fail+=1;return
            sample=_fmt(parsed)
            inst,out=sample.get("instruction","").strip(),sample.get("output","")
            if len(inst)<3 or len(out)<20: fail+=1;return
            if inst in seen: return
            seen.add(inst);gen+=1
            with open(out_path,"a",encoding="utf-8") as f:
                f.write(json.dumps(sample,ensure_ascii=False)+"\n")
        with ThreadPoolExecutor(max_workers=conc) as ex:
            futs=[ex.submit(_one) for _ in range(conc)]
            last=0
            while gen<target and not self._stop:
                while len([f for f in futs if not f.done()])<conc and not self._stop:
                    futs.append(ex.submit(_one))
                futs=[f for f in futs if not f.done()]
                now=time.time()
                if now-last>=1:
                    elapsed=now-t0;rate=gen/max(elapsed,0.1)
                    q.put(("progress",{"gen":gen,"target":target,"fail":fail,"rate":rate,"elapsed":elapsed,"eta":(target-gen)/max(rate,0.01)}))
                    last=now
                time.sleep(0.3)
            if self._stop:
                for f in futs: f.cancel()
        elapsed=time.time()-t0
        q.put(("done",{"gen":gen,"fail":fail,"time":elapsed,"rate":gen/max(elapsed,0.1),"output":str(out_path)}))
        self.running=False
    def stop(self): self._stop=True

engine=GenEngine()

# ============================================================
# Callbacks
# ============================================================
def cb_test(url,key,model):
    c,e=_api(url,key,model,[{"role":"user","content":"回复OK"}],100,0.1)
    if e: return f"FAIL: {e}"
    return f"OK: {c[:80]}"

def cb_models(url,key):
    try:
        r=httpx.get(f"{url.rstrip('/')}/models",headers={"Authorization":f"Bearer {key}"},timeout=10)
        if r.status_code==200:
            return gr.update(choices=[m["id"] for m in r.json().get("data",[])],value="")
    except: pass
    return gr.update(choices=[],value="")

def cb_test_gen(url,key,model,max_tok,temp,jb_name,*tmpl_vals):
    selected=[k for k,v in zip(TEMPLATES.keys(),tmpl_vals) if v]
    if not selected: return "FAIL: 未选模板","",""
    jb_sys=jb_rep=""
    for j in _load(JB_FILE,[]):
        if j["name"]==jb_name: jb_sys=j["system"];jb_rep=j["reply"];break
    rng=random.Random();k=rng.choice(selected);_,tpl,seeds=TEMPLATES[k]
    vars_={k:rng.choice(v) for k,v in seeds.items()};user=tpl
    for k,v in vars_.items(): user=user.replace("{"+k+"}",str(v))
    msgs=[]
    if jb_sys: msgs.append({"role":"system","content":jb_sys})
    if jb_rep: msgs.append({"role":"assistant","content":jb_rep})
    msgs.append({"role":"user","content":f"请生成一条数据集。\n\n{user}\n\n【输出格式】\n<instruction>用户会提出的问题或请求</instruction>\n<input></input>\n<output>详细的回答内容</output>"})
    content,e=_api(url,key,model,msgs,int(max_tok),float(temp))
    if e: return f"FAIL: {e}","",user[:500]
    parsed=_parse(content)
    if parsed:
        s=_fmt(parsed)
        result=f"Q: {s['instruction'][:200]}\n\nA: {s['output'][:600]}"
    else: result=f"(XML parse failed, raw):\n{content[:600]}"
    return "OK","\n---\n".join([f"Q: {s['instruction'][:200]}","A: "+(s.get('output','') or content)[:600]]) if parsed else f"(raw)\n{content[:600]}",user[:500]

def cb_generate(url,key,model,max_tok,temp,conc,target,output,jb_name,*tmpl_vals):
    selected=[k for k,v in zip(TEMPLATES.keys(),tmpl_vals) if v]
    if not selected: yield "FAIL: 未选模板","";return
    jb_sys=jb_rep=""
    for j in _load(JB_FILE,[]):
        if j["name"]==jb_name: jb_sys=j["system"];jb_rep=j["reply"];break
    cfg={"url":url,"key":key,"model":model,"max_tok":max_tok,"temp":temp,"conc":conc,"target":target,"output":output,"jb_sys":jb_sys,"jb_rep":jb_rep,"tmpls":selected}
    q=queue.Queue()
    threading.Thread(target=engine.run,args=(cfg,q),daemon=True).start()
    for _ in range(1800):
        try:
            typ,data=q.get(timeout=1)
            if typ=="progress":
                pct=min(100,int(100*data["gen"]/max(data["target"],1)))
                m,s=divmod(int(data["eta"]),60)
                yield f"RUNNING: {data['gen']}/{data['target']} ({pct}%) | {data['rate']:.1f}/s | fail:{data['fail']} | ETA:{m}m{s}s",(
                    f"<progress value='{pct}' max='100' style='width:100%;height:22px'></progress>"
                    f"<pre style='margin:4px 0'>gen:{data['gen']}/{data['target']} rate:{data['rate']:.1f}/s fail:{data['fail']} elapsed:{data['elapsed']:.0f}s ETA:{m}m{s}s</pre>")
            elif typ=="done":
                m,s=divmod(int(data["time"]),60)
                yield f"DONE: {data['gen']}条 | fail:{data['fail']} | {m}m{s}s | {data['rate']:.1f}/s",""
                return
            elif typ=="error": yield f"ERROR: {data}","";return
        except queue.Empty: continue
    yield "TIMEOUT",""


# ============================================================
# UI
# ============================================================
preset_data=_load(PRESETS_FILE,[])
jb_data=_load(JB_FILE,[])
preset_names=[p["name"] for p in preset_data]
jb_names=[j["name"] for j in jb_data]
tmpl_keys=list(TEMPLATES.keys())

with gr.Blocks(title="数据集生成",theme=gr.themes.Soft()) as app:
    gr.Markdown("# Dataset Generator")

    # ====== SECTION 1: API Preset (collapsible) ======
    with gr.Accordion("API 预设设置", open=False):
        with gr.Row():
            preset_dd=gr.Dropdown(preset_names,label="已保存预设",value=preset_names[0],scale=3)
            btn_load=gr.Button("加载",size="sm",scale=1)
            btn_del=gr.Button("删除",size="sm",variant="stop",scale=1)
        with gr.Row():
            p_name=gr.Textbox(label="预设名称",placeholder="起个名字",scale=2)
            p_jb=gr.Dropdown(jb_names,label="关联破限",value=jb_names[-1],scale=2,allow_custom_value=True)
        with gr.Row():
            p_url=gr.Textbox(label="API地址",placeholder="https://api.xxx.com/v1")
            p_key=gr.Textbox(label="API Key",type="password",placeholder="sk-...")
        with gr.Row():
            p_model=gr.Textbox(label="模型",scale=3)
            btn_models=gr.Button("获取模型列表",size="sm",scale=1)
            btn_test=gr.Button("测试连接",size="sm",scale=1)
        with gr.Row():
            btn_save=gr.Button("保存预设",variant="primary",size="sm")
            p_msg=gr.Textbox(label="",visible=False)

    # ====== SECTION 2: Jailbreak (collapsible) ======
    with gr.Accordion("破限编辑", open=False):
        gr.Markdown("破限版本名用于标识，可复制粘贴到预设的「关联破限」字段中。")
        with gr.Row():
            jb_name=gr.Textbox(label="破限版本名（可复制）",value=jb_data[-1]["name"],scale=2)
            jb_rep=gr.Textbox(label="Fake Reply",value=jb_data[-1]["reply"],scale=3)
        jb_sys=gr.Textbox(label="System Prompt",value=jb_data[-1]["system"],lines=10)
        with gr.Row():
            btn_jb_load=gr.Button("加载选中版本的破限内容",size="sm")
            btn_jb_save=gr.Button("保存破限",variant="primary",size="sm")

    gr.Markdown("---")

    # ====== SECTION 3: Main generation panel ======
    gr.Markdown("## 生成控制台")
    with gr.Row():
        # -- Left: current config --
        with gr.Column(scale=1):
            gr.Markdown("### 当前配置")
            cur_url=gr.Textbox(label="API地址",value=preset_data[0]["url"])
            cur_key=gr.Textbox(label="API Key",value=preset_data[0]["key"],type="password")
            cur_model=gr.Textbox(label="模型",value=preset_data[0]["model"])
            cur_jb=gr.Dropdown(jb_names,label="破限",value=preset_data[0]["jb"],allow_custom_value=True)

            gr.Markdown("### 参数")
            with gr.Row():
                cur_tok=gr.Slider(500,4096,2500,100,label="MaxTokens")
                cur_temp=gr.Slider(0.1,1.5,0.95,0.05,label="Temp")
            with gr.Row():
                cur_conc=gr.Slider(1,50,30,1,label="并发")
                cur_target=gr.Slider(10,10000,2500,10,label="目标数量")
            cur_output=gr.Textbox(label="输出文件（追加模式，不覆盖已有数据）",value="output/generated.jsonl")

            gr.Markdown("### 模板（勾选启用）")
            tmpl_checks={}
            # NSFW group
            gr.Markdown("**NSFW**")
            nsfw_keys=[k for k in tmpl_keys if k not in ("insult1","insult2","general1","general2")]
            with gr.Row():
                for k in nsfw_keys[:3]:
                    label,_,_=TEMPLATES[k];tmpl_checks[k]=gr.Checkbox(label=label,value=True)
            with gr.Row():
                for k in nsfw_keys[3:]:
                    label,_,_=TEMPLATES[k];tmpl_checks[k]=gr.Checkbox(label=label,value=True)
            # Other groups
            gr.Markdown("**其他**")
            other_keys=[k for k in tmpl_keys if k not in nsfw_keys]
            with gr.Row():
                for k in other_keys:
                    label,_,_=TEMPLATES[k];tmpl_checks[k]=gr.Checkbox(label=label,value=False)

        # -- Right: test + generate --
        with gr.Column(scale=1):
            gr.Markdown("### 小批量测试")
            with gr.Row():
                btn_test_gen=gr.Button("生成1条测试",variant="secondary")
                test_status=gr.Textbox(label="",scale=2)
            test_result=gr.Textbox(label="结果预览",lines=12)

            gr.Markdown("---")
            gr.Markdown("### 批量生成")
            with gr.Row():
                btn_gen=gr.Button("开始生成",variant="primary")
                btn_stop=gr.Button("停止",variant="stop")
            gen_status=gr.Textbox(label="状态",value="就绪")
            gen_progress=gr.HTML()

    # ====== Events ======
    def _load_preset(name):
        for p in _load(PRESETS_FILE,[]):
            if p["name"]==name: return p["url"],p["key"],p["model"],p.get("jb","")
        return "","","",""
    btn_load.click(_load_preset,[preset_dd],[cur_url,cur_key,cur_model,cur_jb])
    def _cb_preset_save(n,u,k,m,j):
        if not n.strip(): return "请输入名称"
        ps=_load(PRESETS_FILE,[])
        found=False
        for p in ps:
            if p["name"]==n.strip(): p.update({"url":u,"key":k,"model":m,"jb":j});found=True;break
        if not found: ps.append({"name":n.strip(),"url":u,"key":k,"model":m,"jb":j})
        _save(PRESETS_FILE,ps)
        names=[p["name"] for p in ps]
        return gr.update(choices=names,value=n.strip())
    def _cb_preset_del(n):
        ps=[p for p in _load(PRESETS_FILE,[]) if p["name"]!=n]
        _save(PRESETS_FILE,ps)
        names=[p["name"] for p in ps]
        return gr.update(choices=names,value=names[0] if names else "")
    def _cb_jb_load(n):
        for j in _load(JB_FILE,[]):
            if j["name"]==n: return j["system"],j["reply"]
        return "",""
    def _cb_jb_save(n,s,r):
        if not n.strip(): return "请输入名称"
        jbs=_load(JB_FILE,[]);found=False
        for j in jbs:
            if j["name"]==n.strip(): j.update({"system":s,"reply":r});found=True;break
        if not found: jbs.append({"name":n.strip(),"system":s,"reply":r})
        _save(JB_FILE,jbs)
        return "saved"

    btn_del.click(_cb_preset_del,[preset_dd],[preset_dd])
    btn_save.click(_cb_preset_save,[p_name,p_url,p_key,p_model,p_jb],[preset_dd])
    btn_test.click(cb_test,[p_url,p_key,p_model],[p_msg])
    btn_models.click(cb_models,[p_url,p_key],[p_model])
    btn_jb_load.click(_cb_jb_load,[jb_name],[jb_sys,jb_rep])
    btn_jb_save.click(_cb_jb_save,[jb_name,jb_sys,jb_rep],[p_msg])
    tmpl_inputs=[tmpl_checks[k] for k in tmpl_keys]
    btn_test_gen.click(cb_test_gen,[cur_url,cur_key,cur_model,cur_tok,cur_temp,cur_jb]+tmpl_inputs,[test_status,test_result,gr.Textbox(visible=False)])
    gen_inputs=[cur_url,cur_key,cur_model,cur_tok,cur_temp,cur_conc,cur_target,cur_output,cur_jb]+tmpl_inputs
    btn_gen.click(cb_generate,gen_inputs,[gen_status,gen_progress])
    btn_stop.click(lambda:(engine.stop(),"STOPPED")[1],[],[gen_status])

if __name__=="__main__":
    app.launch(server_name="127.0.0.1",server_port=7860)

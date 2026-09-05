const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const test=require('node:test');
const source=fs.readFileSync(require('node:path').join(__dirname,'..','script.js'),'utf8');
function extract(name){const token=`function ${name}(`;let start=source.indexOf(token);assert.ok(start>=0);if(source.slice(start-6,start)==='async ')start-=6;const end=source.indexOf('\nfunction ',start+token.length);return source.slice(start,end);}
function setup(extra={}){const c={Date,Intl,Map,Set,AbortController,setTimeout,clearTimeout,state:{},...extra};vm.createContext(c);vm.runInContext(['liveResultName','liveResultKey','parseLiveResults'].map(extract).join('\n'),c);return c;}
function bout(extra={}){return {id:'fight',status:{type:{completed:true},period:3,displayClock:'5:00'},competitors:[
  {winner:true,athlete:{fullName:'Nora Cornolle'}},{winner:false,athlete:{fullName:'Klaudia Syguła'}}],
  details:[{type:{text:'Unofficial Winner Decision'}}],...extra};}
const payload=(competitions=[bout()])=>({events:[{id:'event',date:'2026-09-05T16:00Z',competitions}]});
test('confirmed winner and method match both full names and event date',()=>{
  const c=setup(),results=c.parseLiveResults(payload());
  const row=results.get(c.liveResultKey('2026-09-05','Klaudia Sygula','Nora Cornolle'));
  assert.equal(row.winner,'noracornolle');assert.equal(row.method,'DEC');
  assert.equal(results.get(c.liveResultKey('2026-09-12','Nora Cornolle','Klaudia Sygula')),undefined);
  assert.equal(results.get(c.liveResultKey('2026-09-05','Nora Cornolle','Someone Else')),undefined);
});
test('scheduled, drawn, ambiguous and no-contest outcomes never highlight a winner',()=>{
  const c=setup(),key=c.liveResultKey('2026-09-05','Nora Cornolle','Klaudia Sygula');
  for(const b of [bout({status:{type:{completed:false}}}),bout({competitors:bout().competitors.map(p=>({...p,winner:false}))}),bout({details:[{type:{text:'No Contest'}}]})])assert.equal(c.parseLiveResults(payload([b])).get(key),null);
  assert.equal(c.parseLiveResults(payload([bout(),bout()])).has(key),false);
});
test('finish labels use only explicit winner methods, never attempts or knockdowns',()=>{
  const c=setup(),key=c.liveResultKey('2026-09-05','Nora Cornolle','Klaudia Sygula');
  for(const [text,expected] of [['Unofficial Winner Kotko','KO/TKO'],['Unofficial Winner Submission','SUB'],['Submission Attempt',''],['Knockdown','']])
    assert.equal(c.parseLiveResults(payload([bout({details:[{type:{text}}]})])).get(key).method,expected);
});
test('US card midnight rollover matches its local date',()=>{
  const c=setup(),p=payload();p.events[0].date='2026-09-06T01:00Z';
  assert.equal(c.parseLiveResults(p).get(c.liveResultKey('2026-09-05','Nora Cornolle','Klaudia Sygula')).method,'DEC');
});
test('feed failure preserves prior results and backs off without touching predictions',async()=>{
  let calls=0,updates=0;const saved=new Map([['saved',{winner:'a'}]]);
  const c=setup({document:{hidden:false},$:()=>({classList:{contains:()=>true}}),allUpcomingEventGroups:()=>[{event_date:new Date().toISOString().slice(0,10)}],
    fetch:async()=>{calls++;return {ok:false};},applyLiveResultHighlights:()=>updates++});
  vm.runInContext(extract('refreshLiveResults'),c);c.state.liveResults=saved;c.state.model={unchanged:true};
  await c.refreshLiveResults();await c.refreshLiveResults();
  assert.equal(calls,1);assert.equal(updates,1);assert.equal(c.state.liveResults,saved);assert.equal(c.state.liveResultError,true);assert.equal(c.state.model.unchanged,true);
});
test('hidden tabs and unrelated dates make no network request',async()=>{
  let calls=0;const c=setup({document:{hidden:true},$:()=>({classList:{contains:()=>true}}),allUpcomingEventGroups:()=>[{event_date:'2000-01-01'}],fetch:async()=>{calls++;}});
  vm.runInContext(extract('refreshLiveResults'),c);await c.refreshLiveResults();c.document.hidden=false;await c.refreshLiveResults();assert.equal(calls,0);
});

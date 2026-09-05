const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const test = require('node:test');
const source = fs.readFileSync(require('node:path').join(__dirname, '..', 'script.js'), 'utf8');
function extract(name) {const start=source.indexOf(`function ${name}(`);assert.ok(start>=0);return source.slice(start,source.indexOf('\nfunction ',start+1));}
class Node {
  constructor(tag,text='') {this.tag=tag;this.textContent=text;this.children=[];this.style={};this.events={};this.value='';this.attrs={};}
  append(...nodes) {this.children.push(...nodes);}
  replaceChildren(...nodes) {this.children=nodes;}
  setAttribute(k,v) {this.attrs[k]=v;}
  addEventListener(k,fn) {this.events[k]=fn;}
}
const series={matchup_id:'m',event_id:'e',fighter_id:'a',opponent_id:'b',market:'moneyline',selection_id:'a',label:'A',points:[
  {book:'One',observed_at_utc:'2026-09-01T12:00:00Z',moneyline:-110},
  {book:'One',observed_at_utc:'2026-09-02T12:00:00Z',moneyline:120},
  {book:'Two',observed_at_utc:'2026-09-02T12:00:00Z',moneyline:125}]};
function setup() {
  const context={state:{},Date,document:{createElementNS:(_,tag)=>new Node(tag)},element:(tag,cls,text)=>new Node(tag,text),
    appendText:(node,tag,cls,text)=>node.append(new Node(tag,text)),fetchJson:async()=>({series:[series]}),formatOdds:String,formatTimestamp:String};
  vm.createContext(context);vm.runInContext(['finite','decimalOdds','betHistoryMatches','renderBetOddsHistory'].map(extract).join('\n'),context);return context;
}
test('exact matchup, method fighter and totals line matching',()=>{
  const c=setup();assert.ok(c.betHistoryMatches(series,{matchup_id:'m'}));
  assert.equal(c.betHistoryMatches(series,{matchup_id:'other'}),false);
  const total={...series,market:'total_rounds',line:1.5};
  assert.equal(c.betHistoryMatches(total,{matchup_id:'m',category:'Total rounds',line:2.5}),false);
  const method={...series,market:'method',selection_id:'a:ko_tko'};
  assert.ok(c.betHistoryMatches(method,{matchup_id:'m',fighter_id:'a',method:'ko_tko'}));
  assert.equal(c.betHistoryMatches(method,{matchup_id:'m',fighter_id:'b',method:'ko_tko'}),false);
});
test('opening a recommendation renders all books, filter and point timestamps',async()=>{
  const c=setup(), details=c.renderBetOddsHistory({matchup_id:'m',side:'fighter',fighter_id:'a'});
  details.open=true;await details.events.toggle();
  const walk=n=>[n,...n.children.flatMap(walk)];
  assert.equal(walk(details).filter(n=>n.tag==='circle').length,3);
  assert.equal(walk(details).filter(n=>n.tag==='polyline').length,2);
  const selectors=walk(details).filter(n=>n.tag==='select');selectors[1].value='One';selectors[1].events.change();
  assert.equal(walk(details).filter(n=>n.tag==='circle').length,2);
  const point=walk(details).find(n=>n.tag==='circle');point.events.focus();
  assert.ok(walk(details).some(n=>n.textContent.includes('book updated unknown')));
});

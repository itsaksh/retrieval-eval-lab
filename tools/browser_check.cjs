// Optional UI verification: install Playwright separately; it is not a Python dependency.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const path = require('path'), fs = require('fs'), {pathToFileURL} = require('url');
(async()=>{
  const root=path.resolve(__dirname,'..');
  const browser=await chromium.launch({headless:true,...(process.env.CHROME_EXECUTABLE?{executablePath:process.env.CHROME_EXECUTABLE}:{})});
  try {
    const page=await browser.newPage({viewport:{width:1280,height:1000}}), errors=[];
    page.on('pageerror',e=>errors.push(String(e)));
    await page.goto(pathToFileURL(path.join(root,'reports/click-comparison.html')).href);
    const result=await page.evaluate(()=>{
      const data=JSON.parse(document.getElementById('data').textContent);let states=0;
      for(const method of Object.keys(data.methods))for(const budget of data.budgets)
        for(const strategy of ['prefix','whole_chunks'])for(const row of data.methods[method].queries){
          for(const [id,value]of Object.entries({method,budget:String(budget),strategy,query:row.query_id}))document.getElementById(id).value=value;
          document.getElementById('query').dispatchEvent(new Event('change'));
          const expected=row.packs.find(p=>p.budget_tokens===budget&&p.strategy===strategy);
          if(document.getElementById('context').textContent!==(expected.text||'(empty context)'))throw Error('wrong displayed context');
          if(document.getElementById('hits').children.length!==row.hits.length)throw Error('wrong ranking size');
          states++;
        }
      return{states,horizontalOverflow:document.documentElement.scrollWidth>innerWidth};
    });
    await page.selectOption('#method','hybrid_rrf');await page.selectOption('#budget','512');
    await page.selectOption('#strategy','whole_chunks');await page.selectOption('#query','c05');
    await page.screenshot({path:path.join(root,'reports/report-preview.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    const mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
    if(errors.length||result.horizontalOverflow||mobileOverflow)throw Error(JSON.stringify({errors,result,mobileOverflow}));
    const record={browser:'Chromium/Chrome headless',...result,mobileOverflow,javascriptErrors:errors};
    fs.writeFileSync(path.join(root,'reports/browser-check.json'),JSON.stringify(record,null,2)+'\n');
    console.log(JSON.stringify(record,null,2));
  }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});

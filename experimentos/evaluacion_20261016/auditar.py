import json, zipfile, io, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from scipy.fft import dctn

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def features(im):
    im=ImageOps.exif_transpose(im).convert('RGB')
    arr=np.asarray(im)
    ph=dctn(np.asarray(im.resize((32,32)).convert('L'),dtype=float),norm='ortho')[:8,:8].flatten()
    bits=ph>np.median(ph[1:])
    return hashlib.sha256(str(im.size).encode()+arr.tobytes()).hexdigest(),np.packbits(bits).tobytes().hex()

records=[];summary={}
for zpath in sorted((ROOT/'DataSetsDeEntrenamiento').glob('*samples.zip')):
    rows=[];thumbs=[]
    with zipfile.ZipFile(zpath) as z:
        names=[n for n in z.namelist() if Path(n).suffix.lower() in ['.jpg','.jpeg','.png','.webp']]
        for j,n in enumerate(names):
            raw=z.read(n)
            try:
                im=Image.open(io.BytesIO(raw));im.load()
                pix,ph=features(im)
                rows.append(dict(zip=zpath.name,name=n,sha256=hashlib.sha256(raw).hexdigest(),pixels=pix,phash=ph,size=im.size))
                if j in np.linspace(0,len(names)-1,15,dtype=int):
                    thumbs.append((n,ImageOps.contain(im.convert('RGB'),(150,130))))
            except Exception as e: rows.append(dict(zip=zpath.name,name=n,error=str(e)))
    records.extend(rows)
    summary[zpath.stem]=dict(entries=len(names),valid=sum('pixels' in r for r in rows),unique_pixels=len({r['pixels'] for r in rows if 'pixels'in r}),sizes=dict(Counter(str(r.get('size')) for r in rows)))
    sheet=Image.new('RGB',(750,480),'white');draw=ImageDraw.Draw(sheet)
    for j,(n,im) in enumerate(thumbs):
        x=j%5*150;y=j//5*160;sheet.paste(im,(x,y));draw.text((x+2,y+132),n[:22],fill='black')
    sheet.save(OUT/(zpath.stem+'.jpg'))
(OUT/'auditoria_muestras.json').write_text(json.dumps({'resumen':summary,'registros':records},indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))

sources=[]
for folder in [ROOT/'archive',Path.home()/'Downloads'/'sapipise.v1i.folder',Path.home()/'Downloads'/'kambingpise.v1i.folder']:
    count=Counter()
    for p in folder.rglob('*'):
        if p.suffix.lower() not in ['.jpg','.jpeg','.png']:continue
        try:
            im=Image.open(p);im.load();pix,ph=features(im)
            sources.append(dict(source=folder.name,path=str(p),pixels=pix,phash=ph))
            count[p.parent.name]+=1
        except Exception:pass
    print(folder.name,dict(count),flush=True)
(OUT/'fuentes_locales.json').write_text(json.dumps(sources),encoding='utf-8')
for z in summary:
    rs=[r for r in records if r['zip']==z+'.zip' and 'phash'in r]
    matches=Counter()
    for r in rs:
        ds=[(int(r['phash'],16)^int(s['phash'],16)).bit_count() for s in sources]
        j=int(np.argmin(ds))
        if ds[j]<=4: matches[sources[j]['source']]+=1
    print('Similitud pHash <=4 (no confirma procedencia)',z,dict(matches),flush=True)

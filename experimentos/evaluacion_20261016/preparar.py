"""Reserva 30 imágenes por clase y agrupa coincidencias antes de entrenar."""
import sys, json, random, zipfile, shutil
import io
import numpy as np
from PIL import Image,ImageOps
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from comun.imagenes import hash_archivo, hash_pixeles

def main():
    audit=json.loads((OUT/'auditoria_muestras.json').read_text(encoding='utf-8'))
    sources=json.loads((OUT/'fuentes_locales.json').read_text(encoding='utf-8'))
    rows=[r for r in audit['registros'] if 'pixels'in r]
    config=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    labelzip={'PielSana':'PielSana-samples.zip','PielConLSD':'PielConLSD-samples.zip','NoValido':'NoValida-samples.zip','OjoRosado':'OjoRosado-samples.zip','OjoSano':'OjoSano-samples.zip'}
    byzip={labelzip[c['etiqueta']]:c for c in config['clases']}
    parents=list(range(len(rows)))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    def union(a,b):parents[find(b)]=find(a)
    sourceids={};matches=[];source_cache={}
    zips={z:zipfile.ZipFile(ROOT/'DataSetsDeEntrenamiento'/z) for z in labelzip.values()}
    for i,r in enumerate(rows):
        r['clase']=byzip[r['zip']]['etiqueta'];r['carpeta']=byzip[r['zip']]['carpeta']
        distances=[(int(r['phash'],16)^int(s['phash'],16)).bit_count() for s in sources]
        candidates=[j for j,d in enumerate(distances) if d<=4]
        groups=set()
        for j in candidates:
            s=sources[j];p=Path(s['path'])
            # Roboflow augmentation siblings share the part preceding .rf.
            key=s['source']+'/'+p.name.split('.rf.')[0]
            groups.add(key)
            if key in sourceids:union(i,sourceids[key])
            sourceids[key]=i
        r['fuentes_coincidentes']=sorted(groups)
        if candidates and r['clase'] in ('OjoRosado','OjoSano'):
            arr=np.asarray(Image.open(io.BytesIO(zips[r['zip']].read(r['name']))).convert('RGB'),dtype=np.float32)
            scores=[]
            for j in candidates:
                s=sources[j]
                if s['path'] not in source_cache:
                    source_cache[s['path']]=np.asarray(ImageOps.fit(Image.open(s['path']).convert('RGB'),(224,224),method=Image.Resampling.LANCZOS),dtype=np.float32)
                scores.append((float(np.abs(arr-source_cache[s['path']]).mean()),j))
            mae,j=min(scores)
            r['coincidencia_mae']=mae
            r['fuente_verificada']=sources[j]['source'] if mae<=8 else None
            r['fuente_archivo']=sources[j]['path'] if mae<=8 else None
        r['excluir_especie']=r.get('fuente_verificada')=='kambingpise.v1i.folder'
        matches.append(bool(groups))
    for z in zips.values():z.close()
    (OUT/'trazabilidad_muestras.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    # Conservative perceptual grouping joins probable copies, including cross labels.
    for i,r in enumerate(rows):
        for j in range(i):
            if r['pixels']==rows[j]['pixels'] or (int(r['phash'],16)^int(rows[j]['phash'],16)).bit_count()<=4:union(i,j)
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[find(i)].append(r)
    conflicts=[g for g in groups.values() if len({r['clase'] for r in g})>1]
    conflict_ids={id(r) for g in conflicts for r in g}
    (OUT/'conflictos.json').write_text(json.dumps(conflicts,ensure_ascii=False,indent=2),encoding='utf-8')
    if (OUT/'division').exists():raise RuntimeError('La división ya existe; no se sobrescribe.')
    for k,v in config['rutas'].items():config['rutas'][k]=str((ROOT/v).resolve())
    for k,v in [('originales','originales'),('division','division'),('modelo','modelo'),('etiquetas','modelo/labels.txt'),('evaluacion','resultados')]:config['rutas'][k]=str(OUT/v)
    config['modelo']['preprocesamiento']['confirmado']=True
    config['datos']['prueba_independiente_confirmada']=True
    plan=[];excluded=[];counts={};ziphandles={z:zipfile.ZipFile(ROOT/'DataSetsDeEntrenamiento'/z) for z in labelzip.values()}
    for c in config['clases']:
        gs=[[r for r in g if not r['excluir_especie']] for g in groups.values() if g[0]['clase']==c['etiqueta'] and not any(id(r)in conflict_ids for r in g)]
        gs=[g for g in gs if g]
        gs.sort(key=lambda g:min(r['sha256'] for r in g));random.Random(42).shuffle(gs)
        if len(gs)<=30:raise RuntimeError(f'{c["etiqueta"]}: solo {len(gs)} grupos')
        for index,g in enumerate(gs):
            subset='prueba' if index<30 else 'entrenamiento'
            seen=set()
            for r in sorted(g,key=lambda r:(r['sha256'],r['name'])):
                if r['pixels']in seen or (subset=='prueba' and seen):
                    excluded.append(dict(r,motivo='duplicado o variante del grupo reservado'));continue
                seen.add(r['pixels'])
                rel=Path(c['carpeta'])/(r['sha256']+'.jpg');orig=OUT/'originales'/rel
                orig.parent.mkdir(parents=True,exist_ok=True);orig.write_bytes(ziphandles[r['zip']].read(r['name']))
                dest=Path(subset)/rel;target=OUT/'division'/dest;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(orig,target)
                plan.append(dict(clase=c['etiqueta'],carpeta=c['carpeta'],ruta=rel.as_posix(),sha256=hash_archivo(orig),pixeles_sha256=hash_pixeles(orig),conjunto=subset,destino=dest.as_posix(),grupo=min(x['sha256'] for x in g),zip=r['zip'],miembro=r['name']))
        counts[c['etiqueta']]={'entradas':sum(r['clase']==c['etiqueta'] for r in rows),'unicas_pixeles':len({r['pixels'] for r in rows if r['clase']==c['etiqueta']}),'grupos_sin_conflicto':len(gs),'excluidas_cabras':sum(r['clase']==c['etiqueta'] and r['excluir_especie'] for r in rows),'excluidas_conflicto':sum(r['clase']==c['etiqueta'] for g in conflicts for r in g),'entrenamiento':sum(r['clase']==c['etiqueta'] and r['conjunto']=='entrenamiento' for r in plan),'prueba':30,'fuentes_verificadas':dict(Counter(r.get('fuente_verificada','sin metadatos') for r in rows if r['clase']==c['etiqueta']))}
    for z in ziphandles.values():z.close()
    manifest={'creado_utc':datetime.now(timezone.utc).isoformat(),'semilla':42,'prueba_por_clase':30,'registros':plan,'duplicados_excluidos':excluded,'conflictos_excluidos':sum(map(len,conflicts)),'metodo':'Agrupación pHash de 64 bits distancia <=4 y familias de aumento Roboflow; una imagen por grupo de prueba. No acredita independencia por animal.'}
    for name,data in [('config_experimento.json',config),('division/manifiesto.json',manifest),('conteos.json',counts)]:
        (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(counts,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()

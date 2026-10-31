"""Paired GT/prediction overlays; input images remain unchanged."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from shapely.geometry import shape,Polygon
from shared_manifest import overlay


def read(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def main(a):
    rows=read(a.manifest);locations=json.loads(Path(a.locations).read_text())
    pred={p['sample_id']:p for p in read(a.predictions)}
    available={r['source_image_id'] for r in rows
               if 'clean::'+r['source_image_id'] in pred and 'deg::'+r['variant_id'] in pred}
    sources=[]
    if a.worst and Path(a.worst).is_file():
        sources=[r['source_image_id'] for r in json.loads(Path(a.worst).read_text())
                 if r['source_image_id'] in available]
    sources=list(dict.fromkeys(sources+sorted(available)))[:a.limit]
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    cards=[]
    for row in rows:
        if row['source_image_id'] not in sources or 'deg::'+row['variant_id'] not in pred:continue
        for condition in ('clean','degraded'):
            key='clean::'+row['source_image_id'] if condition=='clean' else 'deg::'+row['variant_id']
            path=Path(locations[condition+'_root'])/row[condition+'_relpath']
            with Image.open(path) as im:
                im=im.convert('RGB');drawing=ImageDraw.Draw(im)
                overlay(shape(row['roi_'+condition]),drawing,'#00BFFF',3)
                for item in row['gt_'+condition]['ignored']:overlay(shape(item['geometry']),drawing,'#999999')
                for item in row['gt_'+condition]['valid']:overlay(shape(item['geometry']),drawing,'#00FF00')
                for coordinates in pred[key]['polygons']:
                    pts=np.asarray(coordinates).reshape(-1,2)
                    if len(pts)>=3:overlay(Polygon(pts),drawing,'#FF0000')
                im.thumbnail((1280,1280))
                name=row['source_image_id']+'_'+row['variant_id']+'_'+condition+'.jpg'
                im.save(out/name,quality=88)
                cards.append((key,name))
    import html
    markup='<meta charset="utf-8"><title>Paired detection audit</title><h1>Clean/degraded GT + predictions</h1><p>Green valid GT; gray ignored; cyan valid field of view; red raw predictions.</p>'
    markup+=''.join(f'<figure><figcaption>{html.escape(key)}</figcaption><img style="max-width:100%" src="{html.escape(name)}"></figure>' for key,name in cards)
    (out/'gallery.html').write_text(markup,encoding='utf-8')
    print('Overlays:',len(cards),'in',out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('manifest','locations','predictions','output-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--worst',default='');p.add_argument('--limit',type=int,default=10)
    main(p.parse_args())

"""Download small public reference tables for reproducible local checks."""
import argparse
import hashlib
from pathlib import Path
import urllib.request


def download(directory,extended=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    selections={'wdl':['KQvK','KRvK','KBNvK','KQvKR','KRvKB','KPvKP','KPvK',
                       'KBvK','KNvK','KQvKP','KRvKP','KBvKP','KNvKP','KRvKR','KRvKN',
                       'KNNvK','KQvKN','KBvKN','KNvKN'],
                'dtz':['KQvK','KRvK','KQvKR','KRvKB','KPvKP']}
    if extended:selections['wdl'].append('KNNvKP')
    for metric,names in selections.items():
        for name in names:
            filename=name+('.rtbw' if metric=='wdl' else '.rtbz');path=directory/filename
            if not path.exists():
                url=f'https://tablebase.lichess.ovh/tables/standard/3-4-5-{metric}/{filename}'
                with urllib.request.urlopen(url,timeout=30) as response:data=response.read()
                if len(data)%64!=16:raise ValueError('reference file failed Syzygy size check')
                path.write_bytes(data)
            print(filename,path.stat().st_size,hashlib.sha256(path.read_bytes()).hexdigest(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',default='references');p.add_argument('--extended',action='store_true',help='also fetch the 13.7 MB five-piece cursed/blessed reference')
    args=p.parse_args();download(args.directory,args.extended)

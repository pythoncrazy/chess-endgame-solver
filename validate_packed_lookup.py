"""All-cell checks of selective decoding and authenticated-frame mutations."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_storage import candidates,decode
from packed_lookup import prepare


def validate():
    rng=np.random.default_rng(9918);frames=mutations=0;cells=0
    for n in (0,1,3,8,17,63,64,65,1024):
        arrays=[rng.integers(-2,3,n,dtype=np.int64),np.arange(n,dtype=np.int64),
                rng.integers(-(1<<62),1<<62,n,dtype=np.int64)]
        for x in arrays:
            for frame,_ in candidates(x,allow_lzma=False):
                checksum=hashlib.blake2b(frame,digest_size=16).digest();page=prepare(frame,checksum)
                if page is None:continue
                expected=decode(frame);actual=page.take(np.arange(n));assert np.array_equal(expected,actual)
                assert np.array_equal(x,actual);frames+=1;cells+=n
                for offset in (0,5,len(frame)-1):
                    changed=bytearray(frame);changed[offset]^=1
                    try:prepare(changed,checksum)
                    except ValueError:mutations+=1
                    else:raise AssertionError('corrupt authenticated frame accepted')
                try:page[.5]
                except IndexError:pass
                else:raise AssertionError('noninteger cell accepted')
    return dict(status='pass',frames=frames,cells_checked=cells,
                corrupt_frame_mutations_rejected=mutations,all_selected_cells_match_full_decode=True)


if __name__=='__main__':
    result=validate();Path('packed_lookup_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

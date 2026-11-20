"""Shared font setting for the Section 5 figures: Times New Roman.

Call use_times() after a script has set its own rcParams.  The Times New
Roman files are registered from the usual system locations when present
(on this machine the Windows font folder under WSL).  Without them the
metric-compatible clones TeX Gyre Termes, Nimbus Roman or Liberation Serif
are used.  Mathematics uses the same face, with STIX for symbols it lacks.
"""
from pathlib import Path
import matplotlib
from matplotlib import font_manager as fm
FOLDERS=['/mnt/c/Windows/Fonts','C:/Windows/Fonts','/usr/share/fonts/truetype/msttcorefonts','/Library/Fonts','/System/Library/Fonts/Supplemental',
         '/usr/share/texmf/fonts/opentype/public/tex-gyre','/usr/share/fonts/opentype/urw-base35','/usr/share/fonts/truetype/liberation']
FILES=['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf','Times New Roman.ttf','Times New Roman Bold.ttf','Times New Roman Italic.ttf','Times New Roman Bold Italic.ttf',
       'texgyretermes-regular.otf','texgyretermes-bold.otf','texgyretermes-italic.otf','texgyretermes-bolditalic.otf',
       'NimbusRoman-Regular.otf','NimbusRoman-Bold.otf','NimbusRoman-Italic.otf','NimbusRoman-BoldItalic.otf']
PREFERENCE=['Times New Roman','TeX Gyre Termes','Nimbus Roman','Liberation Serif','STIXGeneral']

def use_times():
    for folder in FOLDERS:
        for name in FILES:
            path=Path(folder)/name
            if path.exists():
                try:fm.fontManager.addfont(str(path))
                except Exception:pass
    known={f.name for f in fm.fontManager.ttflist};face=next(n for n in PREFERENCE if n in known)
    matplotlib.rcParams.update({'font.family':'serif','font.serif':PREFERENCE,'mathtext.fontset':'custom','mathtext.rm':face,
        'mathtext.it':face+':italic','mathtext.bf':face+':bold','mathtext.cal':face+':italic','mathtext.sf':face,'mathtext.tt':face,'mathtext.fallback':'stix','pdf.fonttype':42})
    return face

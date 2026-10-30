"""Génère rapport.html depuis rapport.md avec pandoc et style.css.
Ouvrir ensuite rapport.html dans Chrome ou Edge, Ctrl+P, « Enregistrer au format PDF »,
cocher « Graphiques d'arrière-plan », marges « Par défaut ».
Usage : python build_pdf.py   (depuis le dossier rapport/)"""
import pathlib, subprocess

here = pathlib.Path(__file__).resolve().parent
html = subprocess.run(['pandoc', str(here / 'rapport.md'), '-s', '--metadata', 'title=Rapport',
                       '--metadata', 'lang=fr'], capture_output=True, text=True, encoding='utf-8', check=True).stdout
html = html.replace('<p><strong>', '<p class="lead"><strong>')
html = html.replace('</head>', '<style>' + (here / 'style.css').read_text(encoding='utf-8') + '</style></head>')
(here / 'rapport.html').write_text(html, encoding='utf-8')
print('rapport.html généré')

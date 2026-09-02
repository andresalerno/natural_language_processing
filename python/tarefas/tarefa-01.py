"""Tarefa 01 - PC.1: atributos dos corpora machado, mac_morpho e floresta (NLTK).

Professor: Fabrício Galende Marques de Carvalho
Aluno: André Salerno (RA: 1461392411015)

Versao .py do notebook tarefa-01.ipynb (apenas o codigo da PC.1).
"""

import io
import contextlib

import nltk
from nltk.corpus.reader import CategorizedPlaintextCorpusReader


# =============================================================================
# 0) Setup: garantir os corpora no disco + utilitarios
# =============================================================================
# IMPORTANTE: no NLTK 3.9.1, chamar nltk.download() para um corpus que JA foi
# carregado em memoria corrompe o ponteiro interno do reader (_root perde o
# atributo _path) e qualquer acesso a `corpus.root` passa a lancar
#   AttributeError: 'FileSystemPathPointer' object has no attribute '_path'
# Por isso so baixamos o que ainda nao existe.
def garantir(pacote, recurso):
    try:
        nltk.data.find(recurso)
    except LookupError:
        nltk.download(pacote, quiet=True)


garantir("machado", "corpora/machado")
garantir("mac_morpho", "corpora/mac_morpho")
garantir("floresta", "corpora/floresta")

from nltk.corpus import mac_morpho, floresta  # noqa: E402  (apos garantir o download)


def _raiz(corpus):
    """str do diretorio raiz do corpus, resistente ao bug do _path no NLTK 3.9.1."""
    try:
        return str(corpus.root)
    except AttributeError:
        return "(indisponivel - reinicie o processo)"


def resumo_generico(nome, corpus):
    """Exibe, de forma compacta, os atributos comuns a qualquer CorpusReader."""
    fids = corpus.fileids()        # dispara o carregamento "preguicoso" do reader
    try:
        enc = corpus.encoding(fids[0])
    except Exception:
        enc = "n/d"
    metodos = [m for m in ("raw", "words", "sents", "paras", "tagged_words",
                           "tagged_sents", "parsed_sents", "categories")
               if hasattr(corpus, m)]
    print(f"### {nome}  (reader: {type(corpus).__name__})")
    print("raiz     :", _raiz(corpus))
    print("cod./arq.:", enc, "/", len(fids), "arquivo(s)  ->", fids[:4],
          "..." if len(fids) > 4 else "")
    print("metodos  :", metodos)


# =============================================================================
# (a) MACHADO  --  Obra Completa de Machado de Assis
# =============================================================================
def parte_a_machado():
    # Reconstrucao manual do reader (contorna o bug do loader oficial no NLTK 3.9.1)
    raiz_machado = nltk.data.find("corpora/machado.zip/machado/")
    machado = CategorizedPlaintextCorpusReader(
        raiz_machado, r"(?!\.).*\.txt",
        cat_pattern=r"([a-z]+)/.*",    # categoria = subdiretorio (romance, contos, ...)
        encoding="latin-1",
    )

    resumo_generico("(a) MACHADO", machado)

    print("\ncategorias (generos):")
    for cat in machado.categories():
        fids = machado.fileids(categories=cat)
        print(f"  {cat:11s} {len(fids):3d} arq.  ex.: {fids[0]}")
    print("total de palavras no corpus:",
          sum(len(machado.words(f)) for f in machado.fileids()))

    # README: catalogo com titulo + ANO de publicacao (secao dos romances)
    readme_machado = nltk.data.find(
        "corpora/machado.zip/machado/README").open().read().decode("latin-1")
    romances = readme_machado.split("Romance", 1)[1].split("Poesia", 1)[0].strip()
    print("\n----- README: romances (titulo + ano) -----")
    print(romances)

    # Um DOCUMENTO do corpus: o romance "Dom Casmurro" (1899)
    doc = "romance/marm08.txt"
    print(f"\n----- Documento de exemplo: {doc} -----")
    print(f"paragrafos={len(machado.paras(doc))}  frases={len(machado.sents(doc))}"
          f"  palavras={len(machado.words(doc))}")
    print("metadados (inicio do arquivo):", " ".join(machado.words(doc)[:6]))
    print("frase de exemplo            :", " ".join(machado.sents(doc)[6]))


# =============================================================================
# (b) MAC-MORPHO  --  noticias em portugues do Brasil anotadas com POS
# =============================================================================
def parte_b_mac_morpho():
    resumo_generico("(b) MAC-MORPHO", mac_morpho)

    # O README esta gravado em UTF-8, mas mac_morpho.readme() o decodifica com a
    # codificacao do corpus (latin-1) e gera acentos quebrados ("NAocleo...").
    # Lemos o arquivo manualmente em UTF-8.
    readme_mm = nltk.data.find(
        "corpora/mac_morpho/README").open().read().decode("utf-8")
    print("\n----- README (lido em UTF-8) -----")
    print(readme_mm.strip())

    etiquetas = sorted({tag for _, tag in mac_morpho.tagged_words()})
    gramaticais = [t for t in etiquetas if t[:1].isalpha()]
    print(f"\npalavras={len(mac_morpho.words())}  "
          f"frases={len(mac_morpho.tagged_sents())}  "
          f"etiquetas_distintas={len(etiquetas)}")
    print("etiquetas gramaticais (amostra):", gramaticais[:22])
    print(f"\nfrase etiquetada de exemplo (fileid = {mac_morpho.fileids()[0]}):")
    print(mac_morpho.tagged_sents()[0][:12])


# =============================================================================
# (c) FLORESTA SINTACTICA  --  treebank do portugues (PT europeu + PT-BR)
# =============================================================================
def parte_c_floresta():
    resumo_generico("(c) FLORESTA SINTACTICA", floresta)

    print("\n----- README (cabecalho) -----")
    print(floresta.readme()[:210].rstrip())
    print("   [...]  chave das etiquetas: "
          "http://visl.sdu.dk/visl/pt/portsymbol.html")

    # Percorrer TODA a Floresta imprime mensagens "Bad tree detected..." em
    # sys.stderr: algumas frases do .ptb tem parenteses desbalanceados e o
    # BracketParseCorpusReader as substitui por uma analise sintatica "plana".
    # Capturamos as mensagens so para manter a saida limpa e contamos quantas
    # frases foram recuperadas.
    avisos = io.StringIO()
    with contextlib.redirect_stderr(avisos):
        n_palavras = len(floresta.words())
        frases = floresta.parsed_sents()
        n_frases = len(frases)
        etiquetas = sorted({tag for _, tag in floresta.tagged_words()})
        arvore = frases[0]
    n_recup = avisos.getvalue().count("Bad tree detected")

    print(f"\npalavras={n_palavras}  frases={n_frases}  "
          f"etiquetas_distintas={len(etiquetas)}")
    print(f"frases recuperadas com parse 'plano': {n_recup} "
          f"({n_recup / n_frases:.1%})")
    print("etiquetas 'funcao+categoria' (amostra):",
          [t for t in etiquetas if any(c.isalpha() for c in t)][:8])
    print("\nfrase 0 - tagged:", floresta.tagged_sents()[0])
    print("frase 0 - arvore:", arvore)
    arvore.pretty_print()


if __name__ == "__main__":
    parte_a_machado()
    print()
    parte_b_mac_morpho()
    print()
    parte_c_floresta()

import newspaper
from newspaper import Article
import csv
from spacy.lang.tokenizer_exceptions import URL_PATTERN
import spacy
from spacy.tokens import Span, Doc, Token
from spacy.language import Language
from spacy.tokenizer import Tokenizer
import re


#ALREADY_SEEN_CSV = "lazarobot/articles_index.csv"
NLP = spacy.load('es_core_news_sm', disable=["ner"])



def custom_tokenizer(nlp):
	# contains the regex to match all sorts of urls:
	prefix_re = re.compile(spacy.util.compile_prefix_regex(Language.Defaults.prefixes).pattern.replace("#", "!"))
	infix_re = spacy.util.compile_infix_regex(Language.Defaults.infixes)
	suffix_re = spacy.util.compile_suffix_regex(Language.Defaults.suffixes)

	#special_cases = {":)": [{"ORTH": ":)"}]}
	#prefix_re = re.compile(r'''^[[("']''')
	#suffix_re = re.compile(r'''[])"']$''')
	#infix_re = re.compile(r'''[-~]''')
	#simple_url_re = re.compile(r'''^#''')

	hashtag_pattern = r'''|^(#[\w_-]+)$'''
	url_and_hashtag = URL_PATTERN + hashtag_pattern
	url_and_hashtag_re = re.compile(url_and_hashtag)


	return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
								suffix_search=suffix_re.search,
								infix_finditer=infix_re.finditer,
								token_match=url_and_hashtag_re.match)


def get_article(url):
	try:
		article = Article(url)
		article.download()
		article.parse()
		return  article.text
	except newspaper.article.ArticleException:
		return None


urls = [(1, "https://www.pikaramagazine.com/2018/08/la-industria-de-la-moda-rapida/"), 
		(1, "https://www.pikaramagazine.com/2018/08/yo-queria-sexo-pero-no-asi/"),
		(1, "https://www.pikaramagazine.com/2018/08/las-mujeres-sin-tierra-alimentan-al-mundo-2/"), 
		(1, "https://www.pikaramagazine.com/2018/08/estate-tranquilita-que-ya-bastante-has-hecho-2/"), 
		(1, "https://www.pikaramagazine.com/2018/08/desocupar-la-maternidad-2/"),
		(1, "https://www.pikaramagazine.com/2018/08/la-esclerotica-que-no-encontro-a-su-principe-azul-porque-no-existia/"), 
		(1, "https://www.pikaramagazine.com/2018/08/carta-abierta-a-hombres-feministas/"),
		(1, "https://www.pikaramagazine.com/2018/08/binta-y-la-gran-frontera-2/"),
		(1, "https://www.pikaramagazine.com/2018/08/lo-romantico-es-politico-2/"),
		(1, "https://www.pikaramagazine.com/2018/08/contradicciones-de-una-feminista-en-la-alfombra-roja-2/"),
		(1, "https://www.pikaramagazine.com/2018/08/como-se-folla-bien-2/"),
		(1, "https://www.pikaramagazine.com/2018/07/soy-lola-y-soy-intersexual-2/"),
		(2, "https://www.pikaramagazine.com/2018/08/la-industria-de-la-moda-rapida/"), 
		(2, "https://www.eldiario.es/sociedad/queria-sexo_1_5505365.html"),
		(2, "https://www.pikaramagazine.com/2014/01/las-mujeres-sin-tierra-alimentan-al-mundo/"), 
		(2, "https://www.pikaramagazine.com/2014/09/estate-tranquilita-que-ya-bastante-has-hecho/"), 
		(2, "https://www.pikaramagazine.com/2014/02/desocupar-la-maternidad/"),
		(2, "https://www.pikaramagazine.com/2016/12/esclerosis-multiple-anita-botwin/"), 
		(2, "https://www.pikaramagazine.com/2014/10/una-carta-abierta-a-los-hombres-feministas/"),
		(2, "https://www.pikaramagazine.com/2017/08/binta-y-la-gran-frontera/"),
		(2, "https://www.pikaramagazine.com/2014/02/lo-romantico-es-politico/"),
		(2, "https://www.pikaramagazine.com/2017/02/contradicciones-de-una-feminista-en-la-alfombra-roja/"),
		(2, "https://www.pikaramagazine.com/2013/11/como-se-folla-bien/"),
		(2, "https://www.pikaramagazine.com/2017/03/soy-lola-y-soy-intersexual/"),
		]


if __name__ == "__main__":
	CORPUS_TSV = "corpus/caravaggio_tokenized.tsv"
	NLP.tokenizer = custom_tokenizer(NLP)

	with open(CORPUS_TSV, mode='w', newline='', encoding = "utf-8") as tsv_file:
		file_writer = csv.writer(tsv_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		for tag, url in urls:
			article = get_article(url)
			doc = NLP(article)
			for sent in doc.sents:
				text = " " .join([token.text for token in sent])
				text = re.sub(r'\n+', ' ', text)
				file_writer.writerow([tag, text])
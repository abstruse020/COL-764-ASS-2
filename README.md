###Col-764 Information Retrieval and Web Search
###Assignment - 2

## 1. To Run Rocchio -
bash rocchio rerank.sh [query-file] [top-100-file] [collection-file] [output-
file]
##2. To Run Language Model -
bash lm rerank.sh [rm1|rm2] [query-file] [top-100-file] [collection-dir]
[output-file] [expansions-file]

#Where
rocchio rerank:	bash script file
lm rerank:	bash script file
query-file:	file containing the queries in the same xml form as the training
		queries released
top-100-file:	a file containing the top100 documents in the same format as
		train and dev top100 files given, which need to be reranked
collection-dir:	directory containing the full document collection. Specifically,
		it will have metadata.csv, a subdirectory named docu-
		ment parses which in turn contains subdirectories pdf json
		and pmc json.
output-file:	file to which the output in the trec eval format has to be written
rm1|rm2:	(only for LM) specifies if we are using RM1 or RM2 variant of
		relevance model.
expansions-file:(only for LM) specifis the file to which the expansion terms used
		for each query should be output.


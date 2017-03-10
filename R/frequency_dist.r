library(data.table)
dataset <- NULL
files = dir()
for (name in files) {
	temp_dataset <- read.table(name, header=FALSE, sep="\t")
	colnames(temp_dataset) <- NULL
	dataset <- append(dataset, list(temp_dataset))
}

dataset = rbindlist(dataset)

pal12 = brewer.pal(12, "Paired")
colors = colorRampPalette(pal12)(40) 

# for tag freq distribution log scale
barplot(table(dataset$V4), col=terrain.colors(dataset$V4), log="y", cex.names=0.55, cex.axis = 0.45, xlab="Named-Entity Tags", ylab="Frequency (Log)")

filtered = dataset[V4!="O"]
# filtered has the named entity tags "O" removed from it 
barplot(table(filtered), col=colors, legend=levels(filtered$V2), cex.names=0.8, args.legend = list(x ='topright', bty='n', inset=c(0,0)), xlim=c(0, ncol(filtered) +9), ylab="Frequency", xlab="Named Entity Tag")

# word cloud - word frequency
freqs = count(w)
wordcloud(words = freqs$x.Var1, freq = freqs$x.Freq, min.freq = 1,
	  max.words=500, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))

from Initial import Initial
import codecs

source_file = "/lustre2/sli/leihua/GoogleNewsVector/GoogleNews-vectors-negative300.txt"
target_file = "word2vec"

initial = Initial(all_data_path="xml/all/")
write_file = codecs.open(target_file, mode='w', encoding="utf-8")
index = 0

# temp_dict = set()

with codecs.open(source_file, encoding="utf-8") as openfile:
    for line in openfile:
        index += 1
        if index % 10000 == 0:
            print("current deal with {0}".format(index))

        line = line.strip("\r\n").lstrip().rstrip()
        words = line.split(" ")

        if len(words) != 301:
            continue

        word = words[0]

        if word in initial.word2index:
            # temp_dict.add(word)
            write_file.write(line + "\n")
write_file.close()

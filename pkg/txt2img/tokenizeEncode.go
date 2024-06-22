package txt2img

import (
	"regexp"
	"strings"

	"github.com/davecgh/go-spew/spew"
)

func onNewTokenCb(str *string, bpeTokens *[]int32) bool {
	/*
		wordEnd := strings.Index(*str, ",")
		var embdName string
		if wordEnd == -1 {
			embdName = strings.TrimSpace(*str)
		} else {
			embdName = strings.TrimSpace((*str)[:wordEnd])
		}
		embdPath := getFullPath(embdDir, embdName+".pt")
		if len(embdPath) == 0 {
			embdPath = getFullPath(embdDir, embdName+".ckpt")
		}
		if len(embdPath) == 0 {
			embdPath = getFullPath(embdDir, embdName+".safetensors")
		}
		if len(embdPath) > 0 {
			if loadEmbedding(embdName, embdPath, bpeTokens) {
				if wordEnd != -1 {
					*str = (*str)[wordEnd+1:]
				} else {
					*str = ""
				}
				return true
			}
		}
	*/
	return false
}

func encode(text string) (bpeTokens []int) {
	// originalText := text
	// var bpeTokens []int32
	text = regexp.MustCompile(`\s+`).ReplaceAllString(text, " ")
	text = strings.TrimSpace(text)
	text = strings.ToLower(text)

	spew.Dump(text)

	pattern := `<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[[:alpha:]]+|[[:digit:]]|[^[:space:][:alpha:][:digit:]]+`
	re := regexp.MustCompile(pattern)
	matches := re.FindAllString(text, -1)

	// var tokenStrs []string
	for _, token := range matches {
		// skip := onNewTokenCb(text, bpeTokens)
		// if skip {
		// 	continue
		// }
		// var utf32Token []rune
		// for _, b := range []byte(token) {
		// 	utf32Token = append(utf32Token, byte_encoder[b])
		// }
		// token := bpe(utf32Token)

		spew.Dump(token)

		start := 0
		for pos := strings.Index(token, " "); pos != -1; pos = strings.Index(token[start:], " ") {
			bpeStr := token[start:pos]

			spew.Dump(bpeStr)

			// bpeTokens = append(bpeTokens, encoder[bpeStr])
			// tokenStrs = append(tokenStrs, utf32ToUtf8(bpeStr))
			start = pos + 1
		}
		// bpeStr := token[start:]
		// bpeTokens = append(bpeTokens, encoder[bpeStr])
		// tokenStrs = append(tokenStrs, utf32ToUtf8(bpeStr))
	}

	// fmt.Printf("split prompt \"%s\" to tokens %v\n", originalText, tokenStrs)

	// spew.Dump(originalText)
	// spew.Dump(tokenStrs)

	return
}

/*


func utf32ToUtf8(text []rune) string {
	// Implement UTF-32 to UTF-8 conversion here
	return string(text)
}

var byte_encoder = map[byte]rune{
	// Implement byte to rune mapping here
}

var encoder = map[string]int32{
	// Implement BPE token to integer mapping here
}

func bpe(token string) string {
	word := make([]string, 0)

	for i := 0; i < len(token)-1; i++ {
		word = append(word, string(token[i]))
	}
	word = append(word, token[len(token)-1:]+utf8ToUtf32("</w>"))

	pairs := getPairs(word)

	if len(pairs) == 0 {
		return token + utf8ToUtf32("</w>")
	}

	for {
		minPair := findMinPair(pairs)

		if _, ok := bpeRanks[minPair]; !ok {
			break
		}

		first, second := minPair[0], minPair[1]
		newWord := make([]string, 0)
		i := 0

		for i < len(word) {
			idx := indexOf(word[i:], first)
			if idx == -1 {
				newWord = append(newWord, word[i:]...)
				break
			}
			newWord = append(newWord, word[i:i+idx]...)
			i += idx

			if word[i] == first && i < len(word)-1 && word[i+1] == second {
				newWord = append(newWord, first+second)
				i += 2
			} else {
				newWord = append(newWord, word[i])
				i += 1
			}
		}

		word = newWord

		if len(word) == 1 {
			break
		}
		pairs = getPairs(word)
	}

	result := ""
	for i, w := range word {
		result += w
		if i != len(word)-1 {
			result += " "
		}
	}

	return result
}

func getPairs(word []string) map[string]struct{} {
	pairs := make(map[string]struct{})
	for i := 0; i < len(word)-1; i++ {
		pairs[word[i]+word[i+1]] = struct{}{}
	}
	return pairs
}

func findMinPair(pairs map[string]struct{}) string {
	minPair := ""
	minRank := int64(1 << 63)
	for p := range pairs {
		if _, ok := bpeRanks[p]; !ok {
			continue
		}
		if bpeRanks[p] < minRank {
			minPair = p
			minRank = bpeRanks[p]
		}
	}
	return minPair
}

func indexOf(slice []string, target string) int {
	for i, s := range slice {
		if s == target {
			return i
		}
	}
	return -1
}

func utf8ToUtf32(s string) string {
	// implementation of utf8 to utf32 conversion
	return ""
}
*/

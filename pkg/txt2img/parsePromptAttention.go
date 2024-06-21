package txt2img

import (
	"regexp"
	"strconv"
)

type promptAttention struct {
	Text  string
	Value float64
}

func parsePromptAttention(prompt string) (res []promptAttention) {
	res = make([]promptAttention, 0)

	roundBrackets := make([]int, 0)
	squareBrackets := make([]int, 0)

	roundBracketMultiplier := 1.1
	squareBracketMultiplier := 1 / 1.1

	reAttention := regexp.MustCompile(`\\\(|\\\)|\\\[|\\\]|\\\\|\\|\(|\[|:([+-]?[.\d]+)\)|\)|\]|[^\\()\[\]:]+|:`)
	// reBreak := regexp.MustCompile(`\s*\bBREAK\b\s*`)

	multiplyRange := func(startPosition int, multiplier float64) {
		for p := startPosition; p < len(res); p++ {
			res[p].Value = res[p].Value * multiplier
		}
	}

	remainingText := prompt
	for reAttention.MatchString(remainingText) {
		matches := reAttention.FindStringSubmatch(remainingText)
		text := matches[0]
		weight := matches[1]

		if text == "(" {
			roundBrackets = append(roundBrackets, len(res))
		} else if text == "[" {
			squareBrackets = append(squareBrackets, len(res))
		} else if weight != "" {
			if len(roundBrackets) > 0 {
				multiplyRange(roundBrackets[len(roundBrackets)-1], parseFloat(weight))
				roundBrackets = roundBrackets[:len(roundBrackets)-1]
			}
		} else if text == ")" && len(roundBrackets) > 0 {
			multiplyRange(roundBrackets[len(roundBrackets)-1], roundBracketMultiplier)
			roundBrackets = roundBrackets[:len(roundBrackets)-1]
		} else if text == "]" && len(squareBrackets) > 0 {
			multiplyRange(squareBrackets[len(squareBrackets)-1], squareBracketMultiplier)
			squareBrackets = squareBrackets[:len(squareBrackets)-1]
		} else if text == `\(` {
			res = append(res, promptAttention{
				Text:  text[1:],
				Value: 1.0,
			})
		} else {
			res = append(res, promptAttention{
				Text:  text,
				Value: 1.0,
			})
		}

		remainingText = remainingText[len(matches[0]):]
	}

	for _, pos := range roundBrackets {
		multiplyRange(pos, roundBracketMultiplier)
	}

	for _, pos := range squareBrackets {
		multiplyRange(pos, squareBracketMultiplier)
	}

	if len(res) == 0 {
		res = append(res, promptAttention{
			Text:  "",
			Value: 1.0,
		})
	}

	i := 0
	for i+1 < len(res) {
		if res[i].Value == res[i+1].Value {
			res[i].Text += res[i+1].Text
			res = append(res[:i+1], res[i+2:]...)
		} else {
			i++
		}
	}

	return
}

func parseFloat(s string) float64 {
	f, _ := strconv.ParseFloat(s, 64)
	return f
}

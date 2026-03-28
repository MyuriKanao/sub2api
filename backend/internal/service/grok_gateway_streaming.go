package service

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/google/uuid"
)

// grokStreamLine represents a single line from Grok's NDJSON response.
type grokStreamLine struct {
	Result *grokResult `json:"result,omitempty"`
}

type grokResult struct {
	Response     *grokResponse     `json:"response,omitempty"`
	Conversation *grokConversation `json:"conversation,omitempty"`
	Token        string            `json:"token,omitempty"` // Streaming token (alternative path)
}

type grokResponse struct {
	Token         string             `json:"token,omitempty"`
	ModelResponse *grokModelResponse `json:"modelResponse,omitempty"`
}

type grokModelResponse struct {
	Message    string `json:"message,omitempty"`
	ResponseID string `json:"responseId,omitempty"`
}

type grokConversation struct {
	ConversationID string `json:"conversationId,omitempty"`
}

// OpenAI SSE response types
type openAIChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []openAIChoice `json:"choices"`
}

type openAIChoice struct {
	Index        int          `json:"index"`
	Delta        openAIDelta  `json:"delta"`
	FinishReason *string      `json:"finish_reason"`
}

type openAIDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type openAINonStreamResponse struct {
	ID      string                    `json:"id"`
	Object  string                    `json:"object"`
	Created int64                     `json:"created"`
	Model   string                    `json:"model"`
	Choices []openAINonStreamChoice   `json:"choices"`
	Usage   *openAIUsage              `json:"usage,omitempty"`
}

type openAINonStreamChoice struct {
	Index        int                `json:"index"`
	Message      openAINonStreamMsg `json:"message"`
	FinishReason string             `json:"finish_reason"`
}

type openAINonStreamMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// TransformGrokStreamToOpenAISSE reads Grok's NDJSON response and writes OpenAI SSE format.
func TransformGrokStreamToOpenAISSE(grokBody io.Reader, writer io.Writer, model string) error {
	chatID := "chatcmpl-" + uuid.New().String()[:8]
	created := time.Now().Unix()

	// Send initial role chunk
	roleChunk := openAIChunk{
		ID:      chatID,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []openAIChoice{{
			Index: 0,
			Delta: openAIDelta{Role: "assistant"},
		}},
	}
	if err := grokWriteSSE(writer, roleChunk); err != nil {
		return err
	}

	scanner := bufio.NewScanner(grokBody)
	// Increase scanner buffer for large responses
	scanner.Buffer(make([]byte, 0, 256*1024), 1024*1024)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var grokLine grokStreamLine
		if err := json.Unmarshal([]byte(line), &grokLine); err != nil {
			continue // Skip unparseable lines
		}

		if grokLine.Result == nil {
			continue
		}

		// Extract token from streaming response
		token := ""
		if grokLine.Result.Response != nil {
			token = grokLine.Result.Response.Token
		}
		if token == "" {
			token = grokLine.Result.Token
		}

		if token != "" {
			chunk := openAIChunk{
				ID:      chatID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []openAIChoice{{
					Index: 0,
					Delta: openAIDelta{Content: token},
				}},
			}
			if err := grokWriteSSE(writer, chunk); err != nil {
				return err
			}
		}
	}

	// Send finish chunk
	stopReason := "stop"
	finishChunk := openAIChunk{
		ID:      chatID,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []openAIChoice{{
			Index:        0,
			Delta:        openAIDelta{},
			FinishReason: &stopReason,
		}},
	}
	if err := grokWriteSSE(writer, finishChunk); err != nil {
		return err
	}

	// Send [DONE]
	fmt.Fprintf(writer, "data: [DONE]\n\n")

	return scanner.Err()
}

// TransformGrokToOpenAINonStream reads the full Grok response and returns OpenAI non-stream format.
func TransformGrokToOpenAINonStream(grokBody io.Reader, model string) ([]byte, error) {
	chatID := "chatcmpl-" + uuid.New().String()[:8]
	created := time.Now().Unix()

	scanner := bufio.NewScanner(grokBody)
	scanner.Buffer(make([]byte, 0, 256*1024), 1024*1024)

	var fullContent strings.Builder

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var grokLine grokStreamLine
		if err := json.Unmarshal([]byte(line), &grokLine); err != nil {
			continue
		}

		if grokLine.Result == nil {
			continue
		}

		// Prefer modelResponse.message (final complete response)
		if grokLine.Result.Response != nil && grokLine.Result.Response.ModelResponse != nil {
			if msg := grokLine.Result.Response.ModelResponse.Message; msg != "" {
				fullContent.Reset()
				_, _ = fullContent.WriteString(msg)
			}
		}

		// Accumulate streaming tokens
		token := ""
		if grokLine.Result.Response != nil {
			token = grokLine.Result.Response.Token
		}
		if token == "" {
			token = grokLine.Result.Token
		}
		if token != "" && fullContent.Len() == 0 {
			// Only accumulate if we haven't seen a modelResponse yet
			_, _ = fullContent.WriteString(token)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	content := fullContent.String()
	promptTokens := len(content) / 4 // rough estimate
	completionTokens := len(content) / 4

	resp := openAINonStreamResponse{
		ID:      chatID,
		Object:  "chat.completion",
		Created: created,
		Model:   model,
		Choices: []openAINonStreamChoice{{
			Index: 0,
			Message: openAINonStreamMsg{
				Role:    "assistant",
				Content: content,
			},
			FinishReason: "stop",
		}},
		Usage: &openAIUsage{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      promptTokens + completionTokens,
		},
	}

	return json.Marshal(resp)
}

func grokWriteSSE(w io.Writer, data interface{}) error {
	b, err := json.Marshal(data)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "data: %s\n\n", b)
	return err
}

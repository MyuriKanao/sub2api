package service

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/google/uuid"
)

// geminiEStreamChunk represents a single chunk from Gemini's JSON Array streaming response.
type geminiEStreamChunk struct {
	UToken               string                      `json:"uToken,omitempty"`
	StreamAssistResponse *geminiEStreamAssistResponse `json:"streamAssistResponse,omitempty"`
	Error                *geminiEError                `json:"error,omitempty"`
}

type geminiEStreamAssistResponse struct {
	Answer       *geminiEAnswer      `json:"answer,omitempty"`
	SessionInfo  *geminiESessionInfo `json:"sessionInfo,omitempty"`
	AssistToken  string              `json:"assistToken,omitempty"`
}

type geminiEAnswer struct {
	Name      string           `json:"name,omitempty"`
	State     string           `json:"state,omitempty"` // "IN_PROGRESS" or "SUCCEEDED"
	Replies   []geminiEReply   `json:"replies,omitempty"`
	ADKAuthor string           `json:"adkAuthor,omitempty"`
}

type geminiEReply struct {
	GroundedContent *geminiEGroundedContent `json:"groundedContent,omitempty"`
	ReplyID         string                  `json:"replyId,omitempty"`
}

type geminiEGroundedContent struct {
	Content *geminiEContent `json:"content,omitempty"`
}

type geminiEContent struct {
	Role    string `json:"role,omitempty"`
	Text    string `json:"text,omitempty"`
	Thought bool   `json:"thought,omitempty"`
}

type geminiESessionInfo struct {
	Session string `json:"session,omitempty"`
	QueryID string `json:"queryId,omitempty"`
	TurnID  string `json:"turnId,omitempty"`
}

type geminiEError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
}

// TransformGeminiEStreamToOpenAISSE reads Gemini's JSON Array response and writes OpenAI SSE format.
//
// Gemini response format: JSON Array where each element is a chunk separated by ",\n"
// [{"streamAssistResponse":{"answer":{"state":"IN_PROGRESS","replies":[{"groundedContent":{"content":{"text":"...","thought":true}}}]}}},
//  {"streamAssistResponse":{"answer":{"state":"SUCCEEDED",...}}}]
func TransformGeminiEStreamToOpenAISSE(geminiBody io.Reader, writer io.Writer, model string) error {
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
	if err := geminiEWriteSSE(writer, roleChunk); err != nil {
		return err
	}

	// Read the entire response (Gemini uses JSON Array, not NDJSON)
	bodyBytes, err := io.ReadAll(io.LimitReader(geminiBody, 10*1024*1024)) // 10MB max
	if err != nil {
		return fmt.Errorf("read gemini-e body: %w", err)
	}

	bodyStr := strings.TrimSpace(string(bodyBytes))

	// Parse JSON Array: the response is a JSON array of chunks
	var chunks []geminiEStreamChunk
	if err := json.Unmarshal([]byte(bodyStr), &chunks); err != nil {
		// Try parsing as concatenated JSON objects (comma-separated)
		// Remove leading [ and trailing ]
		inner := strings.TrimPrefix(bodyStr, "[")
		inner = strings.TrimSuffix(inner, "]")

		// Split by },\n{ and wrap each piece
		for _, piece := range splitJSONObjects(inner) {
			piece = strings.TrimSpace(piece)
			if piece == "" {
				continue
			}
			var chunk geminiEStreamChunk
			if json.Unmarshal([]byte(piece), &chunk) == nil {
				chunks = append(chunks, chunk)
			}
		}
	}

	// Process each chunk
	for _, chunk := range chunks {
		// Check for errors
		if chunk.Error != nil {
			errChunk := openAIChunk{
				ID:      chatID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []openAIChoice{{
					Index: 0,
					Delta: openAIDelta{Content: fmt.Sprintf("[Error %d: %s]", chunk.Error.Code, chunk.Error.Message)},
				}},
			}
			geminiEWriteSSE(writer, errChunk)
			break
		}

		if chunk.StreamAssistResponse == nil || chunk.StreamAssistResponse.Answer == nil {
			continue
		}

		answer := chunk.StreamAssistResponse.Answer

		// Skip if SUCCEEDED with no text (final metadata chunk)
		if answer.State == "SUCCEEDED" && len(answer.Replies) == 0 {
			continue
		}

		for _, reply := range answer.Replies {
			if reply.GroundedContent == nil || reply.GroundedContent.Content == nil {
				continue
			}
			content := reply.GroundedContent.Content

			// Skip thinking chunks (thought: true) — or include if you want
			if content.Thought {
				continue
			}

			text := content.Text
			if text == "" {
				continue
			}

			contentChunk := openAIChunk{
				ID:      chatID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []openAIChoice{{
					Index: 0,
					Delta: openAIDelta{Content: text},
				}},
			}
			if err := geminiEWriteSSE(writer, contentChunk); err != nil {
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
	if err := geminiEWriteSSE(writer, finishChunk); err != nil {
		return err
	}

	fmt.Fprintf(writer, "data: [DONE]\n\n")
	return nil
}

// TransformGeminiEToOpenAINonStream reads the full Gemini response and returns OpenAI JSON format.
func TransformGeminiEToOpenAINonStream(geminiBody io.Reader, model string) ([]byte, error) {
	chatID := "chatcmpl-" + uuid.New().String()[:8]
	created := time.Now().Unix()

	bodyBytes, err := io.ReadAll(io.LimitReader(geminiBody, 10*1024*1024))
	if err != nil {
		return nil, fmt.Errorf("read gemini-e body: %w", err)
	}

	var chunks []geminiEStreamChunk
	if err := json.Unmarshal(bodyBytes, &chunks); err != nil {
		inner := strings.TrimPrefix(strings.TrimSpace(string(bodyBytes)), "[")
		inner = strings.TrimSuffix(inner, "]")
		for _, piece := range splitJSONObjects(inner) {
			var chunk geminiEStreamChunk
			if json.Unmarshal([]byte(strings.TrimSpace(piece)), &chunk) == nil {
				chunks = append(chunks, chunk)
			}
		}
	}

	var fullContent strings.Builder
	for _, chunk := range chunks {
		if chunk.Error != nil {
			return nil, fmt.Errorf("gemini-e error %d: %s", chunk.Error.Code, chunk.Error.Message)
		}
		if chunk.StreamAssistResponse == nil || chunk.StreamAssistResponse.Answer == nil {
			continue
		}
		for _, reply := range chunk.StreamAssistResponse.Answer.Replies {
			if reply.GroundedContent == nil || reply.GroundedContent.Content == nil {
				continue
			}
			content := reply.GroundedContent.Content
			if content.Thought || content.Text == "" {
				continue
			}
			fullContent.WriteString(content.Text)
		}
	}

	contentStr := fullContent.String()
	promptTokens := len(contentStr) / 4
	completionTokens := len(contentStr) / 4

	resp := openAINonStreamResponse{
		ID:      chatID,
		Object:  "chat.completion",
		Created: created,
		Model:   model,
		Choices: []openAINonStreamChoice{{
			Index: 0,
			Message: openAINonStreamMsg{
				Role:    "assistant",
				Content: contentStr,
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

// splitJSONObjects splits a string of comma-separated JSON objects.
// Input: {"a":1},\n{"b":2},\n{"c":3}
// Output: ["{"a":1}", "{"b":2}", "{"c":3}"]
func splitJSONObjects(s string) []string {
	var results []string
	depth := 0
	start := 0

	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '{':
			if depth == 0 {
				start = i
			}
			depth++
		case '}':
			depth--
			if depth == 0 {
				results = append(results, s[start:i+1])
			}
		case '"':
			// Skip string contents
			i++
			for i < len(s) && s[i] != '"' {
				if s[i] == '\\' {
					i++ // skip escaped char
				}
				i++
			}
		}
	}
	return results
}

func geminiEWriteSSE(w io.Writer, data interface{}) error {
	b, err := json.Marshal(data)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "data: %s\n\n", b)
	return err
}

package service

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// GrokGatewayService handles Grok API gateway operations.
// It proxies OpenAI-compatible requests to Grok's web chat API using SSO cookies.
type GrokGatewayService struct {
	httpUpstream     HTTPUpstream
	rateLimitService *RateLimitService
}

// NewGrokGatewayService creates a new GrokGatewayService.
func NewGrokGatewayService(
	httpUpstream HTTPUpstream,
	rateLimitService *RateLimitService,
) *GrokGatewayService {
	return &GrokGatewayService{
		httpUpstream:     httpUpstream,
		rateLimitService: rateLimitService,
	}
}

// GrokForwardResult contains the result of forwarding a request to Grok.
type GrokForwardResult struct {
	Response   *http.Response
	StatusCode int
}

const (
	grokChatAPI   = "https://grok.com/rest/app-chat/conversations/new"
	grokOrigin    = "https://grok.com"
	grokReferer   = "https://grok.com/"
	grokUserAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"

	// Static x-statsig-id: base64("e:TypeError: Cannot read properties of undefined (reading 'childNodes')")
	grokStaticStatsigID = "ZTpUeXBlRXJyb3I6IENhbm5vdCByZWFkIHByb3BlcnRpZXMgb2YgdW5kZWZpbmVkIChyZWFkaW5nICdjaGlsZE5vZGVzJyk="
)

// Grok model name → (modelName, modelMode) mapping
var grokModelMap = map[string][2]string{
	"grok-3-auto":                    {"grok-3-auto", "MODEL_MODE_AUTO"},
	"grok-3":                         {"grok-3-auto", "MODEL_MODE_AUTO"},
	"grok-3-fast":                    {"grok-3-fast", "MODEL_MODE_FAST"},
	"grok-4":                         {"grok-4", "MODEL_MODE_EXPERT"},
	"grok-4-mini-thinking":           {"grok-4-mini-thinking-tahoe", "MODEL_MODE_GROK_4_MINI_THINKING"},
	"grok-4-mini-thinking-tahoe":     {"grok-4-mini-thinking-tahoe", "MODEL_MODE_GROK_4_MINI_THINKING"},
}

// grokPayload is the request body for Grok's app-chat API.
type grokPayload struct {
	DeviceEnvInfo           map[string]interface{} `json:"deviceEnvInfo"`
	DisableMemory           bool                   `json:"disableMemory"`
	DisableSearch           bool                   `json:"disableSearch"`
	DisableTextFollowUps    bool                   `json:"disableTextFollowUps"`
	EnableImageGeneration   bool                   `json:"enableImageGeneration"`
	EnableImageStreaming     bool                   `json:"enableImageStreaming"`
	EnableSideBySide        bool                   `json:"enableSideBySide"`
	FileAttachments         []interface{}          `json:"fileAttachments"`
	ForceConcise            bool                   `json:"forceConcise"`
	ForceSideBySide         bool                   `json:"forceSideBySide"`
	ImageAttachments        []interface{}          `json:"imageAttachments"`
	ImageGenerationCount    int                    `json:"imageGenerationCount"`
	IsAsyncChat             bool                   `json:"isAsyncChat"`
	IsReasoning             bool                   `json:"isReasoning"`
	Message                 string                 `json:"message"`
	ModelMode               string                 `json:"modelMode"`
	ModelName               string                 `json:"modelName"`
	ResponseMetadata        map[string]interface{} `json:"responseMetadata"`
	ReturnImageBytes        bool                   `json:"returnImageBytes"`
	ReturnRawGrokInXaiReq   bool                   `json:"returnRawGrokInXaiRequest"`
	SendFinalMetadata       bool                   `json:"sendFinalMetadata"`
	Temporary               bool                   `json:"temporary"`
	ToolOverrides           map[string]interface{} `json:"toolOverrides"`
	WebpageUrls             []interface{}          `json:"webpageUrls"`
}

// buildGrokHeaders builds HTTP headers for Grok API requests.
func buildGrokHeaders(ssoToken string) http.Header {
	h := http.Header{}
	h.Set("Accept", "*/*")
	h.Set("Accept-Encoding", "gzip, deflate, br, zstd")
	h.Set("Accept-Language", "en-US,en;q=0.9")
	h.Set("Baggage", "sentry-environment=production,sentry-release=d6add6fb0460641fd482d767a335ef72b9b6abb8,sentry-public_key=b311e0f2690c81f25e2c4cf6d4f7ce1c")
	h.Set("Content-Type", "application/json")
	h.Set("Origin", grokOrigin)
	h.Set("Priority", "u=1, i")
	h.Set("Referer", grokReferer)
	h.Set("Sec-Ch-Ua", `"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"`)
	h.Set("Sec-Ch-Ua-Mobile", "?0")
	h.Set("Sec-Ch-Ua-Platform", `"Windows"`)
	h.Set("Sec-Fetch-Dest", "empty")
	h.Set("Sec-Fetch-Mode", "cors")
	h.Set("Sec-Fetch-Site", "same-origin")
	h.Set("User-Agent", grokUserAgent)

	// SSO cookie
	cookie := fmt.Sprintf("sso=%s; sso-rw=%s", ssoToken, ssoToken)
	h.Set("Cookie", cookie)

	// Statsig and request IDs
	h.Set("x-statsig-id", grokStaticStatsigID)
	h.Set("x-xai-request-id", uuid.New().String())

	return h
}

// buildGrokPayload converts an OpenAI-style request body into a Grok payload.
func buildGrokPayload(body []byte) (*grokPayload, error) {
	var openaiReq struct {
		Model    string        `json:"model"`
		Messages []interface{} `json:"messages"`
		Stream   *bool         `json:"stream,omitempty"`
	}
	if err := json.Unmarshal(body, &openaiReq); err != nil {
		return nil, fmt.Errorf("parse request: %w", err)
	}

	// Resolve Grok model
	modelName := openaiReq.Model
	modelMode := "MODEL_MODE_AUTO"
	if mapped, ok := grokModelMap[openaiReq.Model]; ok {
		modelName = mapped[0]
		modelMode = mapped[1]
	}

	// Extract messages into a single combined message
	message := extractMessages(openaiReq.Messages)

	return &grokPayload{
		DeviceEnvInfo: map[string]interface{}{
			"darkModeEnabled":  false,
			"devicePixelRatio": 2,
			"screenHeight":     1329,
			"screenWidth":      2056,
			"viewportHeight":   1083,
			"viewportWidth":    2056,
		},
		DisableMemory:         true,
		DisableSearch:         false,
		DisableTextFollowUps:  false,
		EnableImageGeneration: true,
		EnableImageStreaming:   true,
		EnableSideBySide:      true,
		FileAttachments:       []interface{}{},
		ForceConcise:          false,
		ForceSideBySide:       false,
		ImageAttachments:      []interface{}{},
		ImageGenerationCount:  2,
		IsAsyncChat:           false,
		IsReasoning:           false,
		Message:               message,
		ModelMode:             modelMode,
		ModelName:             modelName,
		ResponseMetadata: map[string]interface{}{
			"requestModelDetails": map[string]interface{}{
				"modelId": modelName,
			},
		},
		ReturnImageBytes:      false,
		ReturnRawGrokInXaiReq: false,
		SendFinalMetadata:     true,
		Temporary:             true,
		ToolOverrides:         map[string]interface{}{},
		WebpageUrls:           []interface{}{},
	}, nil
}

// extractMessages converts OpenAI messages array into a single Grok message string.
func extractMessages(messages []interface{}) string {
	var parts []string

	for i, msg := range messages {
		m, ok := msg.(map[string]interface{})
		if !ok {
			continue
		}
		role, _ := m["role"].(string)
		content := extractContent(m["content"])
		if content == "" {
			continue
		}

		// Last user message gets no role prefix (Grok convention)
		isLastUser := false
		if role == "user" {
			isLastUser = true
			for j := i + 1; j < len(messages); j++ {
				if mm, ok := messages[j].(map[string]interface{}); ok {
					if r, _ := mm["role"].(string); r == "user" {
						isLastUser = false
						break
					}
				}
			}
		}

		if isLastUser {
			parts = append(parts, content)
		} else {
			parts = append(parts, fmt.Sprintf("%s: %s", role, content))
		}
	}

	return strings.Join(parts, "\n\n")
}

// extractContent extracts text from OpenAI content (string or array).
func extractContent(content interface{}) string {
	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		var texts []string
		for _, item := range v {
			if m, ok := item.(map[string]interface{}); ok {
				if t, _ := m["type"].(string); t == "text" {
					if text, _ := m["text"].(string); text != "" {
						texts = append(texts, text)
					}
				}
			}
		}
		return strings.Join(texts, "\n")
	default:
		return ""
	}
}

// generateDynamicStatsigID generates a random x-statsig-id (mimics grok2api's dynamic mode).
func generateDynamicStatsigID() string {
	msg := fmt.Sprintf("e:TypeError: Cannot read properties of undefined (reading '%s')", uuid.New().String()[:8])
	return base64.StdEncoding.EncodeToString([]byte(msg))
}

// Forward proxies an OpenAI-compatible request to Grok's web API.
func (s *GrokGatewayService) Forward(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
) (*GrokForwardResult, error) {
	startTime := time.Now()

	ssoToken := account.GetSSOToken()
	if ssoToken == "" {
		return nil, fmt.Errorf("grok account %d has no SSO token", account.ID)
	}

	// Build Grok payload from OpenAI request
	payload, err := buildGrokPayload(body)
	if err != nil {
		return nil, fmt.Errorf("build grok payload: %w", err)
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal grok payload: %w", err)
	}

	// Build request
	req, err := http.NewRequestWithContext(ctx, "POST", grokChatAPI, bytes.NewReader(payloadBytes))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header = buildGrokHeaders(ssoToken)

	// Get proxy URL from account if available
	proxyURL := ""
	if account.ProxyID != nil && *account.ProxyID > 0 {
		proxyURL = account.GetCredential("proxy_url")
	}

	// Execute request
	resp, err := s.httpUpstream.Do(req, proxyURL, account.ID, account.Concurrency)
	if err != nil {
		return nil, fmt.Errorf("upstream request failed: %w", err)
	}

	// Handle non-200 responses
	if resp.StatusCode != 200 {
		bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		resp.Body.Close()

		if s.rateLimitService != nil {
			s.rateLimitService.HandleUpstreamError(ctx, account, resp.StatusCode, resp.Header, bodyBytes)
		}

		return &GrokForwardResult{
			StatusCode: resp.StatusCode,
		}, fmt.Errorf("grok upstream error: %d, body: %s", resp.StatusCode, string(bodyBytes[:min(len(bodyBytes), 200)]))
	}

	_ = startTime // TODO: track latency metrics

	return &GrokForwardResult{
		Response:   resp,
		StatusCode: 200,
	}, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

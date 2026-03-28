package service

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// GeminiEGatewayService handles Gemini Business (Enterprise) API gateway operations.
// It proxies OpenAI-compatible requests to Gemini's widgetStreamAssist API using C_SES/C_OSES cookies.
type GeminiEGatewayService struct {
	httpUpstream     HTTPUpstream
	rateLimitService *RateLimitService

	// Cache XSRF tokens per account (1h TTL)
	xsrfCache sync.Map // accountID → *geminiEXSRFCache
}

type geminiEXSRFCache struct {
	XSRFToken string
	KeyID     string
	ExpiresAt time.Time
}

// NewGeminiEGatewayService creates a new GeminiEGatewayService.
func NewGeminiEGatewayService(
	httpUpstream HTTPUpstream,
	rateLimitService *RateLimitService,
) *GeminiEGatewayService {
	return &GeminiEGatewayService{
		httpUpstream:     httpUpstream,
		rateLimitService: rateLimitService,
	}
}

// GeminiEForwardResult contains the result of forwarding a request to Gemini Business.
type GeminiEForwardResult struct {
	Response   *http.Response
	StatusCode int
}

const (
	geminiEStreamAssistAPI = "https://biz-discoveryengine.googleapis.com/v1alpha/locations/global/widgetStreamAssist"
	geminiEGetOXSRFAPI     = "https://business.gemini.google/auth/getoxsrf"
	geminiEOrigin          = "https://business.gemini.google"
	geminiEReferer         = "https://business.gemini.google/"
	geminiEUserAgent       = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
)

// Gemini-E model mapping (OpenAI model name → Gemini modelId)
var geminiEModelMap = map[string]string{
	// Direct Gemini model names
	"gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
	"gemini-3-flash-preview": "gemini-3-flash-preview",
	"gemini-2.5-pro":         "gemini-2.5-pro",
	"gemini-2.5-flash":       "gemini-2.5-flash",
	// Aliases
	"gemini-pro":   "gemini-3.1-pro-preview",
	"gemini-flash": "gemini-3-flash-preview",
	"gpt-4o":       "gemini-3.1-pro-preview",
	"gpt-4o-mini":  "gemini-3-flash-preview",
}

// geminiEStreamAssistRequest is the request body for widgetStreamAssist.
type geminiEStreamAssistRequest struct {
	ConfigID         string                 `json:"configId"`
	AdditionalParams map[string]string      `json:"additionalParams"`
	StreamAssist     geminiEStreamAssistBody `json:"streamAssistRequest"`
}

type geminiEStreamAssistBody struct {
	Session              string                    `json:"session"`
	Query                geminiEQuery              `json:"query"`
	Filter               string                    `json:"filter"`
	FileIDs              []string                  `json:"fileIds"`
	AnswerGenerationMode string                    `json:"answerGenerationMode"`
	ToolsSpec            map[string]interface{}     `json:"toolsSpec"`
	LanguageCode         string                    `json:"languageCode"`
	UserMetadata         map[string]string          `json:"userMetadata"`
	AssistSkippingMode   string                    `json:"assistSkippingMode"`
	AssistGenerationConfig geminiEGenerationConfig  `json:"assistGenerationConfig"`
}

type geminiEQuery struct {
	Parts []geminiEPart `json:"parts"`
}

type geminiEPart struct {
	Text string `json:"text"`
}

type geminiEGenerationConfig struct {
	ModelID string `json:"modelId"`
}

// stripXSSI removes Google's anti-XSSI prefix )]}' from JSON responses.
func stripXSSI(text string) string {
	if strings.HasPrefix(text, ")]}'") {
		if idx := strings.Index(text, "\n"); idx != -1 {
			return text[idx+1:]
		}
		return text[4:]
	}
	return text
}

// getXSRFToken obtains or returns cached XSRF token for an account.
func (s *GeminiEGatewayService) getXSRFToken(ctx context.Context, account *Account) (string, string, error) {
	// Check cache
	if cached, ok := s.xsrfCache.Load(account.ID); ok {
		c, ok := cached.(*geminiEXSRFCache)
		if ok && time.Now().Before(c.ExpiresAt) {
			return c.XSRFToken, c.KeyID, nil
		}
		s.xsrfCache.Delete(account.ID)
	}

	csesidx := account.GetCredential("csesidx")
	cSes := account.GetCredential("c_ses")
	cOses := account.GetCredential("c_oses")

	if csesidx == "" || (cSes == "" && cOses == "") {
		return "", "", fmt.Errorf("missing csesidx or cookies")
	}

	url := fmt.Sprintf("%s?csesidx=%s", geminiEGetOXSRFAPI, csesidx)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return "", "", err
	}

	// Set cookies
	var cookieParts []string
	if cSes != "" {
		cookieParts = append(cookieParts, fmt.Sprintf("__Secure-C_SES=%s", cSes))
	}
	if cOses != "" {
		cookieParts = append(cookieParts, fmt.Sprintf("__Host-C_OSES=%s", cOses))
	}
	req.Header.Set("Cookie", strings.Join(cookieParts, "; "))
	req.Header.Set("User-Agent", geminiEUserAgent)

	proxyURL := ""
	if account.ProxyID != nil && *account.ProxyID > 0 {
		proxyURL = account.GetCredential("proxy_url")
	}

	resp, err := s.httpUpstream.Do(req, proxyURL, account.ID, account.Concurrency)
	if err != nil {
		return "", "", fmt.Errorf("getoxsrf request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	bodyBytes, err := io.ReadAll(io.LimitReader(resp.Body, 8192))
	if err != nil {
		return "", "", fmt.Errorf("read getoxsrf response: %w", err)
	}

	if resp.StatusCode != 200 {
		return "", "", fmt.Errorf("getoxsrf returned %d: %s", resp.StatusCode, string(bodyBytes[:min(len(bodyBytes), 200)]))
	}

	cleaned := stripXSSI(string(bodyBytes))
	var xsrfResp struct {
		XSRFToken      string `json:"xsrfToken"`
		ExpirationTime string `json:"expirationTime"`
		KeyID          string `json:"keyId"`
	}
	if err := json.Unmarshal([]byte(cleaned), &xsrfResp); err != nil {
		return "", "", fmt.Errorf("parse getoxsrf: %w", err)
	}

	if xsrfResp.XSRFToken == "" || xsrfResp.KeyID == "" {
		return "", "", fmt.Errorf("empty xsrfToken or keyId")
	}

	// Cache for 50 minutes (actual TTL ~1h, leave buffer)
	s.xsrfCache.Store(account.ID, &geminiEXSRFCache{
		XSRFToken: xsrfResp.XSRFToken,
		KeyID:     xsrfResp.KeyID,
		ExpiresAt: time.Now().Add(50 * time.Minute),
	})

	return xsrfResp.XSRFToken, xsrfResp.KeyID, nil
}

// signJWT creates a JWT Bearer token signed with HMAC-SHA256.
// keyID is used as the "kid" header field, hmacSecret is the signing key.
func signJWT(keyID string, csesidx string, hmacSecret string) (string, error) {
	now := time.Now().Unix()

	header := map[string]string{
		"alg": "HS256",
		"typ": "JWT",
		"kid": keyID,
	}
	payload := map[string]interface{}{
		"iss": "https://business.gemini.google",
		"aud": "https://biz-discoveryengine.googleapis.com",
		"sub": fmt.Sprintf("csesidx/%s", csesidx),
		"iat": now,
		"exp": now + 300, // 5 minutes
		"nbf": now,
	}

	headerJSON, _ := json.Marshal(header)
	payloadJSON, _ := json.Marshal(payload)

	headerB64 := base64.RawURLEncoding.EncodeToString(headerJSON)
	payloadB64 := base64.RawURLEncoding.EncodeToString(payloadJSON)

	signingInput := headerB64 + "." + payloadB64

	// Sign with HMAC-SHA256
	mac := hmac.New(sha256.New, []byte(hmacSecret))
	_, _ = mac.Write([]byte(signingInput))
	signature := base64.RawURLEncoding.EncodeToString(mac.Sum(nil))

	return signingInput + "." + signature, nil
}

// buildGeminiEPayload converts an OpenAI-style request into a Gemini widgetStreamAssist payload.
func buildGeminiEPayload(body []byte, configID string, widgetToken string) ([]byte, string, error) {
	var openaiReq struct {
		Model    string        `json:"model"`
		Messages []interface{} `json:"messages"`
	}
	if err := json.Unmarshal(body, &openaiReq); err != nil {
		return nil, "", fmt.Errorf("parse request: %w", err)
	}

	// Resolve model
	modelID := openaiReq.Model
	if mapped, ok := geminiEModelMap[openaiReq.Model]; ok {
		modelID = mapped
	}

	// Extract messages into a single combined text
	message := extractMessages(openaiReq.Messages)

	// Generate session ID
	sessionID := fmt.Sprintf("%d", time.Now().UnixNano())

	req := geminiEStreamAssistRequest{
		ConfigID: configID,
		AdditionalParams: map[string]string{
			"token": widgetToken,
		},
		StreamAssist: geminiEStreamAssistBody{
			Session: fmt.Sprintf("collections/default_collection/engines/agentspace-engine/sessions/%s", sessionID),
			Query: geminiEQuery{
				Parts: []geminiEPart{{Text: message}},
			},
			Filter:               "",
			FileIDs:              []string{},
			AnswerGenerationMode: "NORMAL",
			ToolsSpec: map[string]interface{}{
				"webGroundingSpec":    map[string]interface{}{},
				"toolRegistry":       "default_tool_registry",
				"imageGenerationSpec": map[string]interface{}{},
			},
			LanguageCode: "en",
			UserMetadata: map[string]string{
				"timeZone": "America/New_York",
			},
			AssistSkippingMode: "REQUEST_ASSIST",
			AssistGenerationConfig: geminiEGenerationConfig{
				ModelID: modelID,
			},
		},
	}

	reqBytes, err := json.Marshal(req)
	if err != nil {
		return nil, "", fmt.Errorf("marshal payload: %w", err)
	}

	return reqBytes, modelID, nil
}

const geminiECompleteQueryAPI = "https://biz-discoveryengine.googleapis.com/v1alpha/locations/global/widgetAdvancedCompleteQuery"

// getWidgetToken obtains a uToken by calling widgetAdvancedCompleteQuery.
// The uToken is required as additionalParams.token for widgetStreamAssist.
func (s *GeminiEGatewayService) getWidgetToken(
	ctx context.Context, account *Account,
	configID string, xsrfToken string, jwt string,
) (string, error) {
	// Check cache (uToken changes per response, but we can reuse for ~5 min)
	cacheKey := fmt.Sprintf("utoken_%d", account.ID)
	if cached, ok := s.xsrfCache.Load(cacheKey); ok {
		c, ok := cached.(*geminiEXSRFCache)
		if ok && time.Now().Before(c.ExpiresAt) {
			return c.XSRFToken, nil // Reusing XSRFToken field for uToken
		}
		s.xsrfCache.Delete(cacheKey)
	}

	payload := map[string]interface{}{
		"configId":         configID,
		"additionalParams": map[string]string{"token": xsrfToken},
		"advancedCompleteQueryRequest": map[string]interface{}{
			"query":           "",
			"suggestionTypes": []string{"PEOPLE", "GOOGLE_WORKSPACE", "CONTENT"},
			"userPseudoId":    uuid.New().String(),
		},
	}
	payloadBytes, _ := json.Marshal(payload)

	req, err := http.NewRequestWithContext(ctx, "POST", geminiECompleteQueryAPI, bytes.NewReader(payloadBytes))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+jwt)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", geminiEUserAgent)

	proxyURL := ""
	if account.ProxyID != nil && *account.ProxyID > 0 {
		proxyURL = account.GetCredential("proxy_url")
	}

	resp, err := s.httpUpstream.Do(req, proxyURL, account.ID, account.Concurrency)
	if err != nil {
		return "", fmt.Errorf("completeQuery failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("completeQuery returned %d: %s", resp.StatusCode, string(bodyBytes[:min(len(bodyBytes), 200)]))
	}

	var result struct {
		UToken string `json:"uToken"`
	}
	if err := json.Unmarshal(bodyBytes, &result); err != nil {
		return "", fmt.Errorf("parse uToken: %w", err)
	}
	if result.UToken == "" {
		return "", fmt.Errorf("empty uToken")
	}

	// Cache for 5 minutes
	s.xsrfCache.Store(cacheKey, &geminiEXSRFCache{
		XSRFToken: result.UToken,
		ExpiresAt: time.Now().Add(5 * time.Minute),
	})

	return result.UToken, nil
}

// Forward proxies an OpenAI-compatible request to Gemini Business widgetStreamAssist.
func (s *GeminiEGatewayService) Forward(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
) (*GeminiEForwardResult, error) {
	configID := account.GetCredential("config_id")
	csesidx := account.GetCredential("csesidx")
	if configID == "" || csesidx == "" {
		return nil, fmt.Errorf("gemini-e account %d missing config_id or csesidx", account.ID)
	}

	// Step 1: Get XSRF token (cached)
	xsrfToken, keyID, err := s.getXSRFToken(ctx, account)
	if err != nil {
		return nil, fmt.Errorf("get xsrf: %w", err)
	}

	// Step 2: Sign JWT Bearer (use xsrfToken as HMAC secret, keyID as kid)
	jwt, err := signJWT(keyID, csesidx, xsrfToken)
	if err != nil {
		return nil, fmt.Errorf("sign jwt: %w", err)
	}

	// Step 3: Get widget token (uToken) via widgetAdvancedCompleteQuery
	uToken, err := s.getWidgetToken(ctx, account, configID, xsrfToken, jwt)
	if err != nil {
		// Fallback: try xsrfToken as widget token
		uToken = xsrfToken
	}

	// Step 4: Build payload with uToken
	payloadBytes, _, err := buildGeminiEPayload(body, configID, uToken)
	if err != nil {
		return nil, fmt.Errorf("build payload: %w", err)
	}

	// Step 5: Build request
	req, err := http.NewRequestWithContext(ctx, "POST", geminiEStreamAssistAPI, bytes.NewReader(payloadBytes))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+jwt)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", geminiEUserAgent)
	req.Header.Set("Origin", geminiEOrigin)
	req.Header.Set("Referer", geminiEReferer)
	req.Header.Set("X-Request-Id", uuid.New().String())

	proxyURL := ""
	if account.ProxyID != nil && *account.ProxyID > 0 {
		proxyURL = account.GetCredential("proxy_url")
	}

	// Step 5: Execute
	resp, err := s.httpUpstream.Do(req, proxyURL, account.ID, account.Concurrency)
	if err != nil {
		return nil, fmt.Errorf("upstream request failed: %w", err)
	}

	if resp.StatusCode != 200 {
		bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		_ = resp.Body.Close()

		// Invalidate XSRF cache on auth errors
		if resp.StatusCode == 401 || resp.StatusCode == 403 {
			s.xsrfCache.Delete(account.ID)
		}

		if s.rateLimitService != nil {
			s.rateLimitService.HandleUpstreamError(ctx, account, resp.StatusCode, resp.Header, bodyBytes)
		}

		return &GeminiEForwardResult{
			StatusCode: resp.StatusCode,
		}, fmt.Errorf("gemini-e upstream error: %d, body: %s", resp.StatusCode, string(bodyBytes[:min(len(bodyBytes), 200)]))
	}

	return &GeminiEForwardResult{
		Response:   resp,
		StatusCode: 200,
	}, nil
}

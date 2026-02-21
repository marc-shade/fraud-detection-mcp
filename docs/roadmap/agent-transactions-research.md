# AI Agent-to-Agent Transactions: Deep Research Report

> Compiled February 2026. 60+ sources across payment platforms, security research, detection solutions, academic papers, and market analysis.

## Executive Summary

McKinsey projects AI agents could mediate **$3-5 trillion** in global consumer commerce by 2030. As of early 2026, every major payment processor (Stripe, Visa, Mastercard, PayPal), every major AI lab (OpenAI, Anthropic, Google), and an emerging blockchain ecosystem (Coinbase, Skyfire) have shipped production infrastructure for agent-initiated transactions. Visa reports a **450% increase** in dark web mentions of "AI Agent" for fraud purposes, and Transmit Security projects **up to 500% increases** in fraud losses as detection systems designed for humans fail against agent traffic.

---

## 1. Current State of AI Agent Transactions (2025-2026)

### 1.1 How AI Agents Execute Financial Transactions

**Browser-Based Agents (Computer Use)**
OpenAI's Operator (launched January 2025) and Anthropic's Computer Use allow agents to directly interact with web UIs -- clicking buttons, filling forms, and completing checkouts. OpenAI's Operator requires human confirmation before financial transactions, and both platforms require users to manually provide one-time passcodes for MFA.

**API-First Agent Commerce**
The dominant model emerging is API-based commerce where agents interact with merchant endpoints programmatically, eliminating the UI layer entirely.

**Blockchain/Crypto Agent Wallets**
Coinbase's Agentic Wallets provide wallet infrastructure designed specifically for AI agents with enterprise-grade security and programmable guardrails. Skyfire provides secure wallet access, verifiable agent identity, and an open payment protocol.

### 1.2 Platforms and APIs

**Stripe Agentic Commerce Suite**
- **Agentic Commerce Protocol (ACP)**: Co-developed with OpenAI, Apache 2.0 licensed. First live standard for programmatic commerce flows. Versions from September 2025 through January 2026. Can be RESTful or MCP server.
- **Shared Payment Tokens (SPTs)**: New payment primitive letting agents initiate payments using buyer's permission without exposing credentials.
- **x402 Protocol**: Automated USDC transactions on Base for micropayments.
- Integrated retailers: Coach, Kate Spade, URBN, Revolve, Ashley Furniture, Squarespace, Wix, Etsy, WooCommerce, BigCommerce.

Sources: [Stripe Agentic Commerce](https://stripe.com/blog/agentic-commerce-suite), [ACP GitHub](https://github.com/agentic-commerce-protocol/agentic-commerce-protocol), [Stripe x402](https://crypto.news/stripe-taps-base-ai-agent-x402-payment-protocol-2026/)

**Visa Trusted Agent Protocol (TAP)**
Unveiled October 2025. Uses agent-specific cryptographic signatures that are merchant-and-purpose-specific, time-bound, and cannot be replayed. Each signature includes timestamps, unique session identifier, key identifier, and algorithm identifier. Hundreds of secure agent transactions completed by December 2025 with partners: Anthropic, IBM, Microsoft, Mistral AI, OpenAI, Perplexity, Samsung, Stripe. Open-source on GitHub.

Sources: [Visa TAP](https://usa.visa.com/about-visa/newsroom/press-releases.releaseId.21716.html), [Visa TAP Developer](https://developer.visa.com/capabilities/trusted-agent-protocol/overview), [Visa TAP GitHub](https://github.com/visa/trusted-agent-protocol)

**Mastercard Agent Pay**
Launched April 2025. Introduces Agentic Tokens -- dynamic, cryptographically secure credentials on existing tokenization infrastructure. OpenAI's ChatGPT Instant Checkout uses it. All U.S. Mastercard cardholders enabled by mid-November 2025.

Sources: [Mastercard Agent Pay](https://www.mastercard.com/us/en/business/artificial-intelligence/mastercard-agent-pay.html), [Agentic Token Framework](https://www.mastercard.com/global/en/news-and-trends/stories/2025/agentic-commerce-framework.html)

**PayPal Agent Ready**
Launched October 28, 2025. Instantly unlocks millions of existing PayPal merchants to accept AI-surface payments. Agent Toolkit integrates payments, invoices, disputes, shipment tracking, catalog, subscriptions. GA early 2026.

Sources: [PayPal Launch](https://newsroom.paypal-corp.com/2025-10-28-PayPal-Launches-Agentic-Commerce-Services-to-Power-AI-Driven-Shopping), [PayPal Toolkit](https://developer.paypal.com/community/blog/paypal-agentic-ai-toolkit/)

**Google Agent Payments Protocol (AP2)**
Launched September 2025, Apache 2.0. Core innovation: **Mandates** -- tamper-proof, cryptographically-signed digital contracts serving as verifiable proof of user instructions. Three types: Intent Mandate (human-not-present), Cart Mandate (human-present), Payment Mandate (shared with network). 60+ initial partners including Mastercard, Adyen, PayPal, Coinbase.

Sources: [Google AP2](https://cloud.google.com/blog/products/ai-machine-learning/announcing-agents-to-payments-ap2-protocol), [AP2 Spec](https://ap2-protocol.org/specification/)

### 1.3 Authentication and Authorization Patterns

| Pattern | Used By | Mechanism |
|---------|---------|-----------|
| Cryptographic Agent Signatures | Visa TAP, Cloudflare Web Bot Auth | HTTP Message Signatures with public key crypto |
| Agentic Tokens | Mastercard Agent Pay | Dynamic cryptographic credentials per transaction |
| Shared Payment Tokens | Stripe SPT | Permissioned payment tokens without credential exposure |
| Mandates (Signed Contracts) | Google AP2 | Tamper-proof digital contracts proving user intent |
| OAuth 2.1 + PKCE | MCP Protocol (Anthropic) | Standard OAuth flow for agent authorization to tools |
| Agentic Wallets | Coinbase, Skyfire | Blockchain wallets with spending limits and guardrails |
| ERC-8004 NFT Credentials | Ethereum | On-chain identity, reputation, and validation registries |

**Cloudflare Web Bot Auth** connects these: stable agent identifiers using HTTP Message Signatures with public key crypto. Visa TAP and Mastercard Agent Pay build on it. Cloudflare partnered with Visa, Mastercard, and American Express.

Sources: [Cloudflare Secure Agentic Commerce](https://blog.cloudflare.com/secure-agentic-commerce/)

### 1.4 MCP-Based Payment Tools

**MCP OAuth 2.1 Authorization**: Added March 26, 2025 revision. PKCE critical for agents deployed in environments where storing secrets is difficult. Claude Code ships native OAuth for remote MCP servers.

**x402-mcp (Vercel)**: Integrates x402 payments with MCP servers. HTTP 402 Payment Required status code lets any MCP tool endpoint request payment. Settles as USDC on Base.

**PayPal MCP Server**: Agent-toolkit available as MCP server for payments, invoices, disputes.

**Agentic AI Foundation (AAIF)**: Established December 2025 under Linux Foundation by Anthropic, Block, and OpenAI. Anchored by MCP (Anthropic), goose (Block), and AGENTS.md (OpenAI). Umbrella governance for agentic interoperability.

Sources: [MCP Auth Spec](https://modelcontextprotocol.io/specification/2025-03-26/basic/authorization), [Vercel x402-mcp](https://vercel.com/blog/introducing-x402-mcp-open-protocol-payments-for-mcp-tools), [AAIF](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation)

---

## 2. Unique Fraud Vectors for AI Agent Transactions

### 2.1 Prompt Injection Attacks on Financial Agents

Most dangerous vector. **AiXBT** (AI cryptocurrency trading agent) was compromised in early 2025 -- attacker gained dashboard access, queued malicious commands causing the agent to transfer ~55 ETH (~$105,000). A multinational bank deployed prompt injection defenses, preventing $18M in potential losses.

The **Promptware Kill Chain** (arXiv 2601.09625) describes how prompt injections escalate from reconnaissance to data exfiltration to financial fraud.

Sources: [Obsidian Security](https://www.obsidiansecurity.com/blog/prompt-injection), [Promptware Kill Chain](https://arxiv.org/pdf/2601.09625), [Unit 42](https://unit42.paloaltonetworks.com/agentic-ai-threats/)

### 2.2 Agent Session Smuggling (A2A Protocol Attacks)

Palo Alto Networks Unit 42 discovered **Agent Session Smuggling** -- a malicious remote agent misuses ongoing sessions to inject unauthorized instructions. In their PoC:
1. Malicious "research assistant" tricked financial assistant into revealing system instructions, tool configs, chat history
2. Escalated to **executing unauthorized stock trades**

Sources: [Unit 42](https://unit42.paloaltonetworks.com/agent-session-smuggling-in-agent2agent-systems/)

### 2.3 Agent Impersonation

Without cryptographic verification (Visa TAP, Cloudflare Web Bot Auth), no reliable way to distinguish legitimate from fraudulent agents making the same API calls.

**Moltbook Agent Network** analysis revealed bot-to-bot prompt injection and data leaks in production multi-agent systems.

Sources: [SecurityWeek](https://www.securityweek.com/security-analysis-of-moltbook-agent-network-bot-to-bot-prompt-injection-and-data-leaks/)

### 2.4 Agent-to-Agent Collusion

arXiv 2402.07510 ("Secret Collusion among AI Agents: Multi-Agent Deception via Steganography"):
- Modern steganographic techniques could render collusion hard to detect
- GPT-4 displayed a **capability jump** suggesting continuous monitoring needed
- Standard security approaches are currently effective but may not remain so

Sources: [arXiv 2402.07510](https://arxiv.org/abs/2402.07510)

### 2.5 Automated High-Frequency Micro-Transactions

Visa saw a **25% increase in malicious bot-initiated transactions** over past 6 months, with U.S. experiencing **40% increase** expected to grow. Attackers spin up thousands of targeted operations in minutes.

Sources: [Group-IB](https://www.group-ib.com/blog/the-dark-side-of-automation-and-rise-of-ai-agent/), [Visa Threats](https://corporate.visa.com/en/sites/visa-perspectives/security-trust/the-threats-landscape-of-agentic-commerce.html)

### 2.6 Social Engineering of Agents

- **Malicious tool descriptions** containing hidden prompt payloads
- **Poisoned data sources** returning adversarial content
- **Synthetic merchant storefronts** appearing legitimate to agent evaluation
- Autonomous agents conducting multi-turn scam calls that adapt and evade guardrails

Sources: [Stytch MCP Vulnerabilities](https://stytch.com/blog/mcp-vulnerabilities/), [Visa Threats](https://corporate.visa.com/en/sites/visa-perspectives/security-trust/the-threats-landscape-of-agentic-commerce.html)

### 2.7 Supply Chain Attacks on Agent Tool Integrations

AI agents in SaaS environments granted **ten times more permissions than needed**. Over 60% of large enterprises now deploy autonomous AI agents in production (up from 15% in 2023).

Sources: [Obsidian Security](https://www.obsidiansecurity.com/blog/ai-agent-market-landscape)

### 2.8 Visa's Comprehensive Threat Taxonomy

**450% increase** in dark web posts mentioning "AI Agent" over 6 months. Threats: agent impersonation, synthetic merchants (AI-generated fake businesses with fabricated compliance docs), data tampering in agent-merchant communications, high-velocity automated fraud, synthetic personas.

Sources: [Visa Threats](https://corporate.visa.com/en/sites/visa-perspectives/security-trust/the-threats-landscape-of-agentic-commerce.html)

---

## 3. How Agent Transactions Differ from Human Transactions

### 3.1 The Fundamental Detection Crisis

Transmit Security "Blinded by the Agent" report (July 2025): consumer AI agents are defeating traditional fraud detection. Systems cannot tell the difference between AI operated by legitimate users and AI operated by fraudsters. Up to **500% increases in fraud losses** projected. Fraud teams face **2-3x more operational workload** over next 12-18 months.

Sources: [Transmit Security](https://transmitsecurity.com/blog/blinded-by-the-agent-how-ai-agents-are-disrupting-fraud-detection)

### 3.2 Dimension-by-Dimension Comparison

| Dimension | Human | Agent | Impact |
|-----------|-------|-------|--------|
| Behavioral Biometrics | Keystroke, mouse, touch | None (API calls only) | Traditional biometrics completely fail |
| Transaction Velocity | Minutes between | Milliseconds; 1000x faster | Velocity rules trigger false positives |
| Decision Patterns | Variable, emotional | Deterministic, model-specific | Low variance looks like bot activity |
| Authentication | Passwords, biometrics, MFA | API keys, OAuth, certs | No "human factor" |
| Geographic Signals | Physical location | Cloud server IPs | GeoIP meaningless |
| Temporal Patterns | Sleep/wake, work hours | 24/7, no circadian rhythm | Time-of-day rules irrelevant |
| Device Fingerprinting | Unique browser/device | Cloud servers, headless browsers | Device fingerprinting fails |
| Session Behavior | Browsing, hesitation | Direct API, structured requests | No "journey" to analyze |
| Error Patterns | Typos, corrections | Clean, precise every time | Lack of errors is suspicious |
| Intent Signals | Search history, dwell time | Structured purchase intents | Cannot infer intent from behavior |

### 3.3 The Anti-Fingerprinting Problem

Advanced bot using anti-fingerprinting and headless browsers evaded detection in **93% of cases**.

Sources: [DataDome](https://datadome.co/learning-center/ai-fraud-detection/)

---

## 4. Existing Solutions and Research

### 4.1 BioCatch Connect 2.0

First major behavioral biometrics platform to address agent transactions (November 2025). Collects **3,000+ multi-signal telemetry elements**. Can differentiate between previously unseen agents, known agents, and human users.

Sources: [BioCatch Connect 2.0](https://www.biocatch.com/press-release/biocatch-connect-2.0-delivers-advanced-fraud-and-financial-crime-fighting-capabilities-to-worlds-banks)

### 4.2 Oscilar

Unified platform spanning fraud detection, identity/credit risk, and compliance for agent-led transactions. Works as a "trust stack" with Visa TAP: TAP assures identity, Oscilar confirms actions are safe.

Sources: [Oscilar](https://oscilar.com/blog/visatap)

### 4.3 Stytch Agent Ready

Full-stack fraud prevention for AI agents: OAuth2 CIBA for back-channel confirmations, device fingerprinting, bot detection, fine-grained RBAC, behavioral monitoring.

Sources: [Stytch Agent Ready](https://stytch.com/ai-agent-ready)

### 4.4 DataDome

Real-time identification and classification of AI agents and LLM crawlers. Dynamic trust scores adjusting continuously based on actual behavior.

Sources: [DataDome Agent Trust](https://datadome.co/agent-trust-management/secure-ai-agents/)

### 4.5 Academic Research

| Paper | Key Contribution |
|-------|------------------|
| RAG-Based Fraud Detection (arXiv 2501.15290) | RAG + ASR + LLM agents for real-time phone fraud |
| FraudTransformer (arXiv 2509.23712) | GPT-style architecture for sequential transaction analysis |
| Secret Collusion (arXiv 2402.07510) | Agent collusion via steganography formalization |
| Agentic AI Cybersecurity Survey (arXiv 2601.05293) | Comprehensive threats and defenses |
| Promptware Kill Chain (arXiv 2601.09625) | Prompt injection escalation to financial fraud |
| Explainable AI + Stacking (arXiv 2505.10050) | XAI for interpretable fraud detection |
| Agentic AI in Financial Services (arXiv 2502.05439) | Multi-agent crews for credit card fraud |

### 4.6 Payment Processor Responses

| Processor | Solution | Status |
|-----------|----------|--------|
| Visa | TAP + Cloudflare Web Bot Auth | Production Dec 2025 |
| Mastercard | Agent Pay + Agentic Tokens | Production Nov 2025 |
| Stripe | ACP + SPT + x402 | Production Sep 2025 |
| PayPal | Agent Ready + Agent Toolkit + AP2 | GA early 2026 |
| American Express | Web Bot Auth | In progress |
| Akamai | Visa identity/fraud controls | Production Dec 2025 |
| Worldpay | Visa/Cloudflare TAP | In progress |

### 4.7 Blockchain Agent Identity

**ERC-8004 (Trustless Agents)** live on Ethereum mainnet January 29, 2026. Three on-chain registries:
1. **Identity Registry** -- ERC-721 (NFT) agent identifiers, portable and transferable
2. **Reputation Registry** -- Standard interface for feedback signals
3. **Validation Registry** -- Proof of compliance with stated parameters

Agent global identifier: `{namespace}:{chainId}:{identityRegistry}`.

**Coinbase AgentKit**: Open-source wallet infrastructure with spending limits, compliance checks, gasless trading on Base.

**Skyfire Network**: $9.5M raised (Coinbase Ventures, a16z CSX) for agent payment infrastructure with verifiable identity.

Sources: [ERC-8004](https://eips.ethereum.org/EIPS/eip-8004), [AgentKit](https://github.com/coinbase/agentkit), [Skyfire](https://www.businesswire.com/news/home/20241024532897/en/)

---

## 5. Novel Detection Approaches

### 5.1 Agent Behavioral Fingerprinting

- **Model-specific response patterns**: Different LLMs produce distinguishable output distributions (Claude vs GPT-4 decision patterns)
- **Response timing signatures**: Characteristic latency profiles per model/provider
- **Decision consistency analysis**: Anomalous deviations from established pattern may indicate compromise
- **Tool usage patterns**: How an agent sequences tool calls creates a behavioral profile

BioCatch's 3,000+ telemetry approach is the current state of the art.

Sources: [BioCatch Behavioral LLMs](https://www.biocatch.com/blog/behavioral-large-language-models-the-next-evolution-of-digital-trust)

### 5.2 Agent Chain-of-Custody Verification

Converging pattern: **provenance chain**: Human -> Authorization Mandate -> Agent Identity -> Transaction Signature -> Merchant Verification.

- Visa TAP: Crypto signatures with timestamps, session IDs, key IDs
- Google AP2 Mandates: Tamper-proof signed contracts proving intent
- Mastercard Agentic Tokens: Unique per-transaction traceable credentials
- ERC-8004: On-chain identity and validation registries
- Signet: Composite trust scores resolving in <50ms

Sources: [Signet](https://agentsignet.com/), [AP2 Spec](https://ap2-protocol.org/specification/)

### 5.3 Multi-Agent Transaction Graph Analysis

- **GNN-based models**: 91% accuracy, AUC 0.961 for coordinated fraud rings (Neo4j + GNN)
- **Cross-agent relationship mapping**: Graphs of agent-merchant-agent interactions for suspicious clustering
- **Temporal correlation analysis**: Synchronized patterns across independent agents
- **Oracle multi-agent system**: Specialized AI agents (transaction analysis, user behavior, correlation) coordinated by central orchestrator

Directly relevant to existing `NetworkAnalyzer` class.

Sources: [GNN Fraud Detection with Neo4j](https://www.analyticsvidhya.com/blog/2025/11/gnn-fraud-detection-with-neo4j/), [Oracle Multi-Agent](https://docs.oracle.com/en/solutions/ai-fraud-detection/index.html)

### 5.4 Intent Verification

Core question: **was this transaction actually authorized by a human principal?**

- Google AP2 Intent Mandates for "human-not-present" scenarios
- OAuth2 CIBA for back-channel user approval
- OpenAI Operator requires explicit human confirmation for financial transactions
- Coinbase/Skyfire programmable spending limits

Fraud detection should verify: valid mandate exists, transaction within parameters, agent identity cryptographically verified, behavior consistent with history.

### 5.5 Agent Capability Boundary Enforcement

Obsidian Security: AI agents granted **10x more permissions than needed**. Requirements:
- Automated token rotation (24-72 hours)
- Enterprise identity provider integration (SAML/OIDC)
- Centralized secret management
- Hardware-backed key storage
- Continuous monitoring against capability profiles
- **AI Security Posture Management (AI-SPM)** as new discipline

Sources: [Obsidian AI-SPM](https://www.obsidiansecurity.com/blog/ai-security-posture-management)

### 5.6 Real-Time Agent Reputation Systems

- **Signet**: Composite trust scores (reliability, quality, financial behavior, security, stability). <50ms lookups.
- **ERC-8004 Reputation Registry**: On-chain feedback signals with on-chain and off-chain scoring.
- **mTrust Protocol**: Proxy between agents and tools, verifying identity and calculating real-time trust.
- **DataDome**: Continuous trust adjustment based on actual behavior.

Robust reputation needs: identity verification (one-time) + behavioral consistency (continuous) + transaction history (accumulated) + peer attestation (cross-platform).

Sources: [Signet](https://agentsignet.com/), [mTrust](https://www.modeltrust.io/), [AgentTrust.AI](https://agenttrust.ai/)

---

## 6. Complete Source List

### Payment Platforms
- [Stripe Agentic Commerce Suite](https://stripe.com/blog/agentic-commerce-suite)
- [Stripe ACP Open Standard](https://stripe.com/blog/developing-an-open-standard-for-agentic-commerce)
- [ACP GitHub](https://github.com/agentic-commerce-protocol/agentic-commerce-protocol)
- [Stripe ACP Specification](https://docs.stripe.com/agentic-commerce/protocol/specification)
- [Stripe Agent Docs](https://docs.stripe.com/agents)
- [Visa TAP](https://usa.visa.com/about-visa/newsroom/press-releases.releaseId.21716.html)
- [Visa TAP Developer](https://developer.visa.com/capabilities/trusted-agent-protocol/overview)
- [Visa TAP GitHub](https://github.com/visa/trusted-agent-protocol)
- [Visa Secure AI 2026](https://usa.visa.com/about-visa/newsroom/press-releases.releaseId.21961.html)
- [Mastercard Agent Pay](https://www.mastercard.com/us/en/business/artificial-intelligence/mastercard-agent-pay.html)
- [Mastercard Agentic Tokens](https://www.mastercard.com/global/en/news-and-trends/stories/2025/agentic-commerce-framework.html)
- [PayPal Agentic Launch](https://newsroom.paypal-corp.com/2025-10-28-PayPal-Launches-Agentic-Commerce-Services-to-Power-AI-Driven-Shopping)
- [PayPal Agent Toolkit](https://developer.paypal.com/community/blog/paypal-agentic-ai-toolkit/)
- [PayPal.ai](https://paypal.ai/)
- [Google AP2](https://cloud.google.com/blog/products/ai-machine-learning/announcing-agents-to-payments-ap2-protocol)
- [AP2 Specification](https://ap2-protocol.org/specification/)
- [Coinbase Agentic Wallets](https://www.coinbase.com/developer-platform/discover/launches/agentic-wallets)
- [Coinbase AgentKit](https://github.com/coinbase/agentkit)
- [x402 GitHub](https://github.com/coinbase/x402)
- [x402 Docs](https://docs.cdp.coinbase.com/x402/welcome)
- [x402 Whitepaper](https://www.x402.org/x402-whitepaper.pdf)
- [Skyfire Network](https://www.businesswire.com/news/home/20241024532897/en/)

### Security and Fraud Research
- [Unit 42 Agent Session Smuggling](https://unit42.paloaltonetworks.com/agent-session-smuggling-in-agent2agent-systems/)
- [Unit 42 Agentic AI Threats](https://unit42.paloaltonetworks.com/agentic-ai-threats/)
- [Visa Agentic Commerce Threats](https://corporate.visa.com/en/sites/visa-perspectives/security-trust/the-threats-landscape-of-agentic-commerce.html)
- [Visa Fraud Risk Flags](https://www.digitalcommerce360.com/2025/11/21/visa-flags-fraud-risks-agentic-commerce/)
- [Transmit Security Blinded by the Agent](https://transmitsecurity.com/blog/blinded-by-the-agent-how-ai-agents-are-disrupting-fraud-detection)
- [Obsidian Security AI Agent Landscape](https://www.obsidiansecurity.com/blog/ai-agent-market-landscape)
- [Obsidian Security Agent Defenses](https://www.obsidiansecurity.com/blog/security-for-ai-agents)
- [Stytch AI Agent Fraud](https://stytch.com/blog/ai-agent-fraud/)
- [Stytch MCP Vulnerabilities](https://stytch.com/blog/mcp-vulnerabilities/)
- [Group-IB Automation Dark Side](https://www.group-ib.com/blog/the-dark-side-of-automation-and-rise-of-ai-agent/)
- [SecurityWeek Moltbook](https://www.securityweek.com/security-analysis-of-moltbook-agent-network-bot-to-bot-prompt-injection-and-data-leaks/)

### Detection Solutions
- [BioCatch Connect 2.0](https://www.biocatch.com/press-release/biocatch-connect-2.0-delivers-advanced-fraud-and-financial-crime-fighting-capabilities-to-worlds-banks)
- [BioCatch Agentic AI](https://www.biocatch.com/blog/agentic-ai-the-next-wave-of-attacks)
- [Oscilar Visa TAP](https://oscilar.com/blog/visatap)
- [Stytch Agent Ready](https://stytch.com/ai-agent-ready)
- [DataDome Agent Trust](https://datadome.co/agent-trust-management/secure-ai-agents/)
- [Signet Agent Trust](https://agentsignet.com/)
- [mTrust Protocol](https://www.modeltrust.io/)

### Infrastructure and Standards
- [Cloudflare Secure Agentic Commerce](https://blog.cloudflare.com/secure-agentic-commerce/)
- [MCP Authorization Spec](https://modelcontextprotocol.io/specification/2025-03-26/basic/authorization)
- [MCP OAuth Analysis](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/)
- [AAIF Linux Foundation](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation)
- [ERC-8004 EIP](https://eips.ethereum.org/EIPS/eip-8004)
- [Vercel x402-mcp](https://vercel.com/blog/introducing-x402-mcp-open-protocol-payments-for-mcp-tools)

### Academic Papers
- [Secret Collusion (arXiv 2402.07510)](https://arxiv.org/abs/2402.07510)
- [Promptware Kill Chain (arXiv 2601.09625)](https://arxiv.org/pdf/2601.09625)
- [Agentic AI Cybersecurity Survey (arXiv 2601.05293)](https://arxiv.org/html/2601.05293v1)
- [RAG Fraud Detection (arXiv 2501.15290)](https://arxiv.org/html/2501.15290v1)
- [FraudTransformer (arXiv 2509.23712)](https://arxiv.org/html/2509.23712v1)
- [Agentic AI Financial Services (arXiv 2502.05439)](https://arxiv.org/html/2502.05439v1)

### Market Analysis
- [McKinsey $3-5T Forecast](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-agentic-commerce-opportunity-how-ai-agents-are-ushering-in-a-new-era-for-consumers-and-merchants)
- [IBM Agentic Commerce](https://www.ibm.com/think/topics/agentic-commerce)
- [CoinGecko Agent Payment Infrastructure](https://www.coingecko.com/learn/ai-agent-payment-infrastructure-crypto-and-big-tech)
- [Agentic Payments Comparison](https://orium.com/blog/agentic-payments-acp-ap2-x402)

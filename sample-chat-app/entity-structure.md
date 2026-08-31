# 1. API Fundamentals

## What is an API?

` API (Application Programming Interface) allows one software application to communicate with another.`

#### Think of a restaurant:

<div style="background-color: #f6f8fae5; border: 1px solid #d0d7de; border-radius: 6px; padding: 12px; display: flex; justify-content: left; align-items: center; min-height: 300px; width: 300px; color: #000000;">

  <table style="border-collapse: collapse; text-align: left; color: #020202; border: 1px solid #000000">
    <thead>
      <tr style="border: 1px solid #706d6d;">
        <th style="padding: 12px 24px; font-weight: bold;">Role</th>
        <th style="padding: 12px 24px; font-weight: bold;">API Equivalent</th>
      </tr>
    </thead>
    <tbody>
      <tr style="border: 1px solid #000000;">
        <td style="padding: 12px 24px;">Customer</td>
        <td style="padding: 12px 24px;">Application/User</td>
      </tr>
      <tr style="border: 1px solid #000000;">
        <td style="padding: 12px 24px;">Waiter</td>
        <td style="padding: 12px 24px;">API</td>
      </tr>
      <tr style="border: 1px solid #000000;">
        <td style="padding: 12px 24px;">Kitchen</td>
        <td style="padding: 12px 24px;">AI Model</td>
      </tr>
      <tr>
        <td style="padding: 12px 24px;">Food</td>
        <td style="padding: 12px 24px;">Response</td>
      </tr>
    </tbody>
  </table>

</div>

### Process

```
 User Request
      │
      ▼
Application
      │
      ▼
API Call
      │
      ▼
AI Model
      │
      ▼
Generated Response
      │
      ▼
Application
```

## REST API Concepts

### HTTP Methods

```
GET     -> Retrieve data
POST    -> Send data
PUT     -> Update data
DELETE  -> Remove data
```

## Request structure

```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "user",
      "content": "Explain AI"
    }
  ]
}
```

#### Response structure

```json
{
  "choices": [
    {
      "message": {
        "content": "Artificial Intelligence is..."
      }
    }
  ]
}
```

# 3. Chat Completion APIs

## Message roles

### System role

```json
{
  "role": "system",
  "content": "You are a helpful teacher."
}
```

### User role

```json
{
  "role": "user",
  "content": "Explain machine learning"
}
```

### Assistant role

```json
{
  "role": "assistant",
  "content": "Machine learning is..."
}
```

### Conversation Example

```
System:
You are a technical trainer.

User:
What is Data Science?

Assistant:
Data Science is ...

User:
Give examples.
```

# 3. System prompts

### Without system prompt

#### user

```
Tell me about cloud computing.
```

#### Response

```
Generic explanation
```

### With System Prompt

#### System

```
You are a senior cloud architect with 15 years of experience.
Explain concepts using real-world examples.
```

#### USer

```
Tell me about cloud computing.
```

#### Response

```
Architect-level explanation
with practical examples.
```

### Examples

### Teacher Assistant

```
You are a patient teacher.
Explain concepts in simple language.
```

#### Coding Assistant

```
You are a Python expert.
Provide code examples.
```

#### Interview Coach

```
You are a senior technology interviewer.
Ask scenario-based questions.
```

# Points to remember

```
✅ Role definition

✅ Output format

✅ Tone specification

✅ Constraints
```

### example

```
You are a data architect.

Rules:
1. Respond in bullet points
2. Maximum 200 words
3. Include architecture diagram in ASCII format
```

# 4. Temperature and Tokens

## Temperature

### Controls creativity

#### Range 0.0 --> 2.0

### Low Temperature

#### temperature=0.1

### Behavior:

- Deterministic
- Consistent
- Fact-focused

#### Useful for:

- Code generation
- Data extraction
- Summarization

### Medium Temperature

#### temperature=0.7

### Behavior:

- Balanced

### Useful for:

- Chatbots
- Business writing

### High Temperature

#### temperature=1.5

### Behavior:

- Creative
- Diverse

### Useful for:

- Story generation
- Marketing ideas

### Temperature Example

#### Prompt

```
Suggest a product name
```

#### Temperature 0.1

#### Response

```
SmartAI
```

## Tokens

`Tokens represent pieces of text.`

### ≈ 2-4 Tokens

---

### Max Tokens

Controls maximum response size.

```python
# Set the maximum number of tokens to generate in the output
max_tokens = 200
```

#### Response

```
Response limited to ~200 tokens
```

#### Benefits:

- Cost control
- Performance optimization
- Prevent overly long answers

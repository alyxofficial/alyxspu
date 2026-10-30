// Alyx I100 SPU firmware for ESP32 (Arduino core). Serial 115200, line-based.
// Commands: PING | INFO | CALC <expr> | NET <in> <hid> <out> | DATA x.. | y.. | CLEAR
//           TRAIN <epochs> <lr> | RUN x.. | STOP not needed (training is short bursts)
#include <Arduino.h>
#include <math.h>

#define MI 8
#define MH 16
#define MO 4
#define MS 32

int nI = 0, nH = 0, nO = 0, ns = 0;
float W1[MH][MI], biasH[MH], W2[MO][MH], biasO[MO];
float X[MS][MI], Y[MS][MO], hh[MH], oo[MO];

// ---------- neural network (1 hidden layer, sigmoid, SGD) ----------
float sig(float x) { return 1.f / (1.f + expf(-x)); }
float rnd() { return (esp_random() / 4294967295.0f) * 2.f - 1.f; }

void initNet() {
  for (int j = 0; j < nH; j++) { biasH[j] = 0; for (int i = 0; i < nI; i++) W1[j][i] = rnd(); }
  for (int k = 0; k < nO; k++) { biasO[k] = 0; for (int j = 0; j < nH; j++) W2[k][j] = rnd(); }
}

void fwd(const float *x) {
  for (int j = 0; j < nH; j++) { float s = biasH[j]; for (int i = 0; i < nI; i++) s += W1[j][i] * x[i]; hh[j] = sig(s); }
  for (int k = 0; k < nO; k++) { float s = biasO[k]; for (int j = 0; j < nH; j++) s += W2[k][j] * hh[j]; oo[k] = sig(s); }
}

float trainEpoch(float lr) {
  float loss = 0, d2[MO], d1[MH];
  for (int s = 0; s < ns; s++) {
    fwd(X[s]);
    for (int k = 0; k < nO; k++) { float e = oo[k] - Y[s][k]; loss += e * e; d2[k] = e * oo[k] * (1 - oo[k]); }
    for (int j = 0; j < nH; j++) { float t = 0; for (int k = 0; k < nO; k++) t += d2[k] * W2[k][j]; d1[j] = t * hh[j] * (1 - hh[j]); }
    for (int k = 0; k < nO; k++) { for (int j = 0; j < nH; j++) W2[k][j] -= lr * d2[k] * hh[j]; biasO[k] -= lr * d2[k]; }
    for (int j = 0; j < nH; j++) { for (int i = 0; i < nI; i++) W1[j][i] -= lr * d1[j] * X[s][i]; biasH[j] -= lr * d1[j]; }
  }
  return loss / (ns * nO);
}

// ---------- calculator (+ - * / ^, parentheses, functions) ----------
const char *p; bool perr;
float expr();
void ws() { while (*p == ' ') p++; }
float prim() {
  ws();
  if (*p == '(') { p++; float v = expr(); ws(); if (*p == ')') p++; else perr = true; return v; }
  if (*p == '-') { p++; return -prim(); }
  if (isalpha(*p)) {
    char n[8]; int i = 0;
    while (isalpha(*p) && i < 7) n[i++] = *p++;
    n[i] = 0; ws();
    if (!strcmp(n, "pi")) return M_PI;
    if (*p != '(') { perr = true; return 0; }
    p++; float v = expr(); ws(); if (*p == ')') p++; else perr = true;
    if (!strcmp(n, "sqrt")) return sqrtf(v);
    if (!strcmp(n, "sin")) return sinf(v);
    if (!strcmp(n, "cos")) return cosf(v);
    if (!strcmp(n, "tan")) return tanf(v);
    if (!strcmp(n, "exp")) return expf(v);
    if (!strcmp(n, "ln")) return logf(v);
    if (!strcmp(n, "abs")) return fabsf(v);
    perr = true; return 0;
  }
  char *e; float v = strtof(p, &e);
  if (e == p) { perr = true; return 0; }
  p = e; return v;
}
float powr() { float b = prim(); ws(); if (*p == '^') { p++; return powf(b, powr()); } return b; }
float term() { float v = powr(); for (;;) { ws(); if (*p == '*') { p++; v *= powr(); } else if (*p == '/') { p++; v /= powr(); } else return v; } }
float expr() { float v = term(); for (;;) { ws(); if (*p == '+') { p++; v += term(); } else if (*p == '-') { p++; v -= term(); } else return v; } }

// ---------- command handling ----------
int parseFloats(const char *s, float *out, int maxn, const char **end) {
  int n = 0;
  while (n < maxn) {
    while (*s == ' ') s++;
    if (!*s || *s == '|') break;
    char *e; float v = strtof(s, &e);
    if (e == s) break;
    out[n++] = v; s = e;
  }
  if (end) *end = s;
  return n;
}

void handle(String line) {
  line.trim();
  if (!line.length()) return;
  int sp = line.indexOf(' ');
  String cmd = sp < 0 ? line : line.substring(0, sp);
  String arg = sp < 0 ? "" : line.substring(sp + 1);
  cmd.toUpperCase();

  if (cmd == "PING") Serial.println("OK ALYX-I100 v1");
  else if (cmd == "INFO") Serial.printf("OK net=%d-%d-%d samples=%d heap=%u\n", nI, nH, nO, ns, ESP.getFreeHeap());
  else if (cmd == "CALC") {
    p = arg.c_str(); perr = false; float v = expr(); ws();
    if (perr || *p) Serial.println("ERR bad expression"); else Serial.printf("RES %.6g\n", v);
  } else if (cmd == "NET") {
    float v[3];
    if (parseFloats(arg.c_str(), v, 3, nullptr) != 3 || v[0] < 1 || v[0] > MI || v[1] < 1 || v[1] > MH || v[2] < 1 || v[2] > MO) {
      Serial.printf("ERR NET in(1-%d) hidden(1-%d) out(1-%d)\n", MI, MH, MO); return;
    }
    nI = v[0]; nH = v[1]; nO = v[2]; ns = 0; initNet();
    Serial.printf("OK net %d-%d-%d\n", nI, nH, nO);
  } else if (cmd == "CLEAR") { ns = 0; Serial.println("OK data cleared"); }
  else if (cmd == "DATA") {
    if (!nI) { Serial.println("ERR run NET first"); return; }
    if (ns >= MS) { Serial.printf("ERR max %d samples\n", MS); return; }
    float a[MI], b[MO]; const char *e;
    int na = parseFloats(arg.c_str(), a, MI, &e);
    if (*e == '|') e++;
    int nb = parseFloats(e, b, MO, nullptr);
    if (na != nI || nb != nO) { Serial.printf("ERR need %d inputs | %d outputs\n", nI, nO); return; }
    memcpy(X[ns], a, sizeof(float) * nI); memcpy(Y[ns], b, sizeof(float) * nO); ns++;
    Serial.printf("OK samples=%d\n", ns);
  } else if (cmd == "TRAIN") {
    float v[2];
    if (!nI || !ns) { Serial.println("ERR need NET and DATA"); return; }
    if (parseFloats(arg.c_str(), v, 2, nullptr) != 2) { Serial.println("ERR TRAIN <epochs> <lr>"); return; }
    int ep = v[0], step = max(1, ep / 50); float loss = 0;
    for (int e = 1; e <= ep; e++) {
      loss = trainEpoch(v[1]);
      if (e % step == 0 || e == ep) Serial.printf("LOSS %d %.6f\n", e, loss);
      if (e % 200 == 0) delay(1);
    }
    Serial.printf("OK trained loss=%.6f\n", loss);
  } else if (cmd == "RUN") {
    float x[MI];
    if (!nI || parseFloats(arg.c_str(), x, MI, nullptr) != nI) { Serial.println("ERR need NET and matching inputs"); return; }
    fwd(x); Serial.print("OUT");
    for (int k = 0; k < nO; k++) Serial.printf(" %.4f", oo[k]);
    Serial.println();
  } else Serial.println("ERR unknown command");
}

void setup() { Serial.begin(115200); delay(300); Serial.println("OK ALYX-I100 ready"); }

void loop() {
  static String buf;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') { handle(buf); buf = ""; }
    else if (c != '\r' && buf.length() < 200) buf += c;
  }
}

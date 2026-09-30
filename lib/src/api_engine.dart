import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

import 'decisions.dart';

/// Answers through any HTTP provider of the Decision API wire format:
/// `POST {"model", "state", "questions"}` → `{"answers": {...}}`, bearer key.
class ApiEngine implements DecisionEngine {
  ApiEngine({
    required this.endpoint,
    this.apiKey,
    this.model,
    this.headers = const {},
    this.extra = const {},
    this.timeout = const Duration(seconds: 60),
    http.Client? client,
  }) : _client = client ?? http.Client(),
       _ownsClient = client == null;

  final Uri endpoint;
  final String? apiKey;
  final String? model;
  final Map<String, String> headers;

  /// Extra top-level fields sent with every request (provider options).
  final Map<String, dynamic> extra;
  final Duration timeout;
  final http.Client _client;
  final bool _ownsClient;

  /// Usage reported by the provider for the last request (for example `{"cost": 0.0004}`), when it sends one.
  Map<String, dynamic>? lastUsage;

  @override
  Future<Map<String, Answer>> decide({Object? state, required Map<String, Question> questions}) async {
    final body = jsonEncode({
      if (model != null) 'model': model,
      'state': state,
      'questions': questions.map((k, q) => MapEntry(k, q.toJson())),
      ...extra,
    });
    final res = await _client
        .post(
          endpoint,
          headers: {
            'Content-Type': 'application/json',
            if (apiKey != null) 'Authorization': 'Bearer $apiKey',
            ...headers,
          },
          body: body,
        )
        .timeout(timeout);
    if (res.statusCode < 200 || res.statusCode >= 300) throw DecisionApiException(res.statusCode, res.body);
    final json = jsonDecode(utf8.decode(res.bodyBytes)) as Map<String, dynamic>;
    lastUsage = json['usage'] as Map<String, dynamic>?;
    final answers =
        json['answers'] as Map<String, dynamic>? ??
        (throw DecisionApiException(res.statusCode, 'no "answers" in response'));
    return {for (final k in questions.keys) k: Answer.fromJson((answers[k] as Map).cast<String, dynamic>())};
  }

  @override
  Future<void> close() async {
    if (_ownsClient) _client.close();
  }
}

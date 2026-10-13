{{- define "radar.fullname" -}}
{{- if contains .Chart.Name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "radar.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}

{{- define "radar.selector" -}}
app.kubernetes.io/name: {{ .root.Chart.Name }}
app.kubernetes.io/instance: {{ .root.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{- define "radar.image" -}}
{{ .Values.image.repository }}:{{ .Values.image.tag }}
{{- end -}}

{{/* Environment of the processes that read the marts with the read-only account. */}}
{{- define "radar.readerEnv" -}}
- name: CLICKHOUSE_URL
  value: {{ .Values.clickhouse.url | quote }}
- name: CLICKHOUSE_RO_USER
  valueFrom:
    secretKeyRef:
      name: {{ .Values.secretName }}
      key: CLICKHOUSE_RO_USER
- name: CLICKHOUSE_RO_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ .Values.secretName }}
      key: CLICKHOUSE_RO_PASSWORD
- name: REDIS_URL
  value: {{ .Values.redis.url | quote }}
{{- with .Values.studentUrl }}
- name: STUDENT_LLM_URL
  value: {{ . | quote }}
{{- end }}
{{- if .Values.salaryModel.enabled }}
- name: SALARY_MODEL_PATH
  value: /models/salary/salary.pkl
{{- end }}
{{- end -}}

{{/* The init container that pulls the salary model by alias, and the volume it fills. */}}
{{- define "radar.fetchModel" -}}
- name: fetch-salary-model
  image: {{ include "radar.image" . }}
  imagePullPolicy: {{ .Values.image.pullPolicy }}
  command:
    - python
    - -m
    - radar.monitor.fetch_model
    - --model
    - {{ .Values.salaryModel.registeredModel | quote }}
    - --alias
    - {{ .Values.salaryModel.alias | quote }}
    - --out
    - /models/salary
  env:
    - name: MLFLOW_TRACKING_URI
      value: {{ .Values.mlflow.trackingUri | quote }}
  securityContext:
    {{- toYaml .Values.securityContext | nindent 4 }}
  resources:
    requests:
      cpu: 50m
      memory: 64Mi
    limits:
      memory: 256Mi
  volumeMounts:
    - name: models
      mountPath: /models
{{- end -}}

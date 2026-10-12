import React, { useMemo } from 'react';
import { AlertCircle, Bot, Check, Facebook, RefreshCw, Share2, Youtube } from 'lucide-react';
import KeyInput from './KeyInput';

const providerLabel = (id) => ({ bedrock: 'Amazon Bedrock', gemini: 'Google Gemini' }[id] || id);

function normalizeAiConfig(value, geminiApiKey, localLlm) {
  const providerList = Array.isArray(value?.providers) ? value.providers : null;
  const order = value?.order || value?.providerOrder || value?.fallbackOrder || providerList?.map((p) => p.name) || ['bedrock', 'gemini'];
  const retries = Number(value?.retriesPerProvider || value?.maxAttemptsPerProvider || value?.retries || 3);
  const providers = providerList ? Object.fromEntries(providerList.map((p) => [p.name, p])) : (value?.providers || {});
  return {
    order,
    retries,
    providers: order.map((id) => {
      const raw = providers[id] || {};
      const configured = typeof raw === 'boolean' ? raw : (raw.configured ?? raw.available);
      return {
        id,
        model: typeof raw === 'object' ? raw.model : null,
        configured: configured ?? (id === 'gemini' ? !!geminiApiKey : false),
      };
    }),
    localLlm,
  };
}

function platformEntries(capabilities) {
  const raw = capabilities?.platforms;
  if (Array.isArray(raw)) {
    return raw.map((item) => typeof item === 'string'
      ? { id: item, configured: true }
      : { id: item.id || item.platform, ...item });
  }
  if (raw && typeof raw === 'object') {
    return Object.entries(raw).map(([id, value]) => typeof value === 'boolean'
      ? { id, configured: value }
      : { id, ...value });
  }
  return [];
}

const PlatformIcon = ({ id }) => {
  if (id === 'youtube') return <Youtube size={17} />;
  if (id === 'facebook' || id === 'instagram') return <Facebook size={17} />;
  return <Share2 size={17} />;
};

export default function IntegrationSettings({
  aiProviders,
  geminiApiKey,
  localLlm,
  onGeminiKeySet,
  socialCapabilities,
  socialLoading,
  onRefreshSocial,
}) {
  const ai = useMemo(
    () => normalizeAiConfig(aiProviders, geminiApiKey, localLlm),
    [aiProviders, geminiApiKey, localLlm],
  );
  const platforms = platformEntries(socialCapabilities);
  const direct = socialCapabilities?.backend === 'direct';

  return (
    <div className="space-y-8">
      <div className="card p-4 sm:p-6">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-input bg-paper3 flex items-center justify-center text-brass"><Bot size={17} /></div>
            <div>
              <h2 className="text-base font-medium text-ink lowercase">AI provider fallback</h2>
              <p className="text-xs text-muted mt-0.5">The server tries providers in this order.</p>
            </div>
          </div>
          <span className="readout">{ai.retries} attempts each</span>
        </div>

        <div className="space-y-2">
          {ai.providers.map((provider, index) => (
            <div key={provider.id} className="rounded-input border border-rule px-3 py-3 flex items-center gap-3">
              <span className="readout w-6 text-center">{index + 1}</span>
              <div className="min-w-0 flex-1">
                <p className="text-sm text-ink font-medium">{providerLabel(provider.id)}</p>
                <p className="text-xs text-muted truncate">{provider.model || (provider.id === 'bedrock' ? 'AWS credentials from the server environment' : 'Server key or browser key below')}</p>
              </div>
              <span className={provider.configured ? 'badge-ok' : 'badge-warn'}>
                {provider.configured ? <><Check size={11} /> ready</> : 'not configured'}
              </span>
            </div>
          ))}
          {ai.localLlm && (
            <div className="rounded-input border border-rule px-3 py-3 text-xs text-muted">
              Local OpenAI-compatible fallback: <span className="text-ink2">{ai.localLlm.model || ai.localLlm.baseUrl || 'configured'}</span>
            </div>
          )}
        </div>
        <p className="text-xs text-muted mt-4 leading-relaxed">
          Bedrock credentials stay in the backend environment (AWS role or access keys). They are never exposed to the browser.
          If all three Bedrock attempts fail, the same request is tried up to three times with Gemini.
        </p>
      </div>

      <KeyInput onKeySet={onGeminiKeySet} savedKey={geminiApiKey} />

      <div className="card p-4 sm:p-6">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-input bg-paper3 flex items-center justify-center text-brass"><Share2 size={17} /></div>
            <div>
              <h2 className="text-base font-medium text-ink lowercase">Publishing destinations</h2>
              <p className="text-xs text-muted mt-0.5">Only destinations enabled by this server are shown.</p>
            </div>
          </div>
          <button onClick={onRefreshSocial} disabled={socialLoading} className="btn-quiet py-1.5 px-3 text-xs">
            <RefreshCw size={13} className={socialLoading ? 'animate-spin' : ''} /> Refresh
          </button>
        </div>

        {platforms.length ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {platforms.map((platform) => (
              <div key={platform.id} className="rounded-input border border-rule px-3 py-3 flex items-center gap-3">
                <span className={platform.configured ? 'text-ok' : 'text-muted'}><PlatformIcon id={platform.id} /></span>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-ink capitalize">{platform.label || platform.id}</p>
                  <p className="text-xs text-muted">{platform.account || platform.channel || (platform.configured ? 'Credentials configured on server' : 'Needs server credentials')}</p>
                </div>
                {platform.configured ? <Check size={15} className="text-ok" /> : <AlertCircle size={15} className="text-warn" />}
              </div>
            ))}
          </div>
        ) : (
          <div className="rounded-input border border-rule px-3 py-3 flex items-start gap-2 text-xs text-muted">
            <AlertCircle size={15} className="text-warn shrink-0 mt-0.5" />
            The backend has not reported a configured publishing destination yet. Add the YouTube OAuth token to the deployment, then refresh.
          </div>
        )}

        <p className="text-xs text-muted mt-4 leading-relaxed">
          {direct
            ? 'Direct publishing is active. Videos and credentials do not pass through a third-party publishing service.'
            : 'This installation is using a compatibility publishing backend. Set SOCIAL_PUBLISH_BACKEND=direct to use the built-in integrations.'}
        </p>
      </div>
    </div>
  );
}

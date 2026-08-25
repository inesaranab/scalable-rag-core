-- NOT ACTIVE: loaded by nginx.conf (access_by_lua_file) once the gateway
-- is deployed on AKS. No Python code imports or runs this.
-- Edge rate limiting, run inside the gateway (Nginx/OpenResty or Kong).
-- Stops abusive traffic BEFORE it reaches the Python API: the same
-- INCR-then-EXPIRE counter as the app-level limiter, but keyed by client
-- address, spending gateway cycles instead of API cycles.
-- The API keeps its own per-user limiter as the second layer.

local redis = require "resty.redis"
local red = redis:new()
red:set_timeout(100)

local ok, err = red:connect("redis", 6379)
if not ok then
    -- A dead Redis must not take the API down with it: let traffic pass.
    return
end

-- Behind a load balancer or ingress, remote_addr is the PROXY's address,
-- so every user would share one bucket. The real client is the first
-- entry of X-Forwarded-For, which nginx also exposes as
-- ngx.var.http_x_forwarded_for. Nginx's real_ip module should be
-- configured with the trusted proxy range so remote_addr is rewritten;
-- until it is, this reads the header directly and falls back.
local forwarded = ngx.var.http_x_forwarded_for
local client = forwarded and forwarded:match("^%s*([^,]+)") or ngx.var.remote_addr

local key = "rate_limit:" .. client
local limit = 100 -- placeholder; set from measured p99 request/min per IP
-- once traffic exists

-- incr returns nil plus an error when Redis fails mid-request; comparing
-- nil to a number raises, which would 500 every request — the opposite
-- of the fail-open intent above.
local current, incr_err = red:incr(key)
if not current then
    red:set_keepalive(10000, 100)
    return
end

if current == 1 then
    red:expire(key, 60)
end

-- Return the connection to the pool instead of abandoning it to garbage
-- collection: a new TCP handshake per request defeats the point of
-- filtering cheaply at the edge.
red:set_keepalive(10000, 100)

if current > limit then
    ngx.status = 429
    ngx.say("Rate limit exceeded.")
    return ngx.exit(429)
end

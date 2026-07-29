"use strict";

const net = require("node:net");

const listenHost = process.argv[2];
const listenPort = Number(process.argv[3] ?? 8000);
const targetHost = process.argv[4] ?? "127.0.0.1";
const targetPort = Number(process.argv[5] ?? 8000);

if (!listenHost || !Number.isInteger(listenPort) || !Number.isInteger(targetPort)) {
  console.error(
    "usage: node scripts/lan-relay.cjs <lan-address> [listen-port] [target-address] [target-port]",
  );
  process.exit(2);
}

process.title = `shrimp-lan-relay-${listenPort}`;

const server = net.createServer((client) => {
  const upstream = net.connect({ host: targetHost, port: targetPort });

  client.pipe(upstream);
  upstream.pipe(client);

  client.on("error", () => upstream.destroy());
  upstream.on("error", () => client.destroy());
});

server.on("error", (error) => {
  console.error(error);
  process.exit(1);
});

server.listen(listenPort, listenHost, () => {
  console.log(
    `Shrimp LAN relay listening on ${listenHost}:${listenPort} -> ${targetHost}:${targetPort}`,
  );
});

